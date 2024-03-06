import json
import pickle
import sys,os
import time
from numpy.matrixlib.defmatrix import matrix
from Process.dataset import sibling_edge
sys.path.append(os.getcwd())
from Process.process import * 
#from Process.process_user import *
import torch as th
import torch
import torch.nn as nn
from torch_scatter import scatter_mean
import torch.nn.functional as F
import numpy as np
from others.earlystopping import EarlyStopping
from torch_geometric.data import DataLoader
from supergat_conv import SuperGATConv
from tqdm import tqdm
from Process.rand5fold import *
from others.evaluate import *
from torch_geometric.nn import GCNConv
import copy
import random
from torch_scatter import scatter_mean,scatter_add
from torch_geometric.utils import softmax
from transformers import BertModel
import warnings
from torch_geometric.data import Data
warnings.filterwarnings("ignore")
def setup_seed(seed):
     th.manual_seed(seed)
     th.cuda.manual_seed_all(seed) 
     np.random.seed(seed)
     random.seed(seed)
     th.backends.cudnn.deterministic = True
my_seed = sys.argv[1]
setup_seed(int(3))
print(my_seed)
label2id = {
            "unverified": 0,
            "non-rumor": 1,
            "true": 2,
            "false": 3,
            }

class GatingModule(nn.Module):
    def __init__(self, input_size):
        super(GatingModule, self).__init__()
        self.W_g = nn.Linear(input_size, 1)
        self.U_g = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, hx, hc,x_local=None):
        g_cx = self.sigmoid(self.W_g(hx) + self.U_g(hc))
        refined_hx = g_cx * hx + (1 - g_cx) * hc
        return refined_hx

def dict_to_tensor(inputs_dict):
    tensor_dict = {k: th.tensor(v) if not th.is_tensor(v) else v for k, v in inputs_dict.items()}
    return tensor_dict


class GCN_Net(th.nn.Module): 
    def __init__(self,in_feats,hid_feats,out_feats): 
        super(GCN_Net, self).__init__() 
        self.conv1 = SuperGATConv(in_feats, hid_feats, heads=8,
                                  dropout=0.6, attention_type='MX',
                                  edge_sample_ratio=0.9, neg_sample_ratio=0.5,is_undirected=False,concat=False)
        self.conv2 = SuperGATConv(hid_feats, out_feats, heads=8,
                                  dropout=0.6, attention_type='MX',
                                  edge_sample_ratio=0.9, neg_sample_ratio=0.5,is_undirected=False,concat=True)
        self.fc=th.nn.Linear(out_feats,4)
        self.out_feats = out_feats
        self.gate3 = GatingModule(hid_feats)
        self.att_pool = nn.Linear(out_feats*2,1)
        self.conv3 = nn.Linear(8, 1)
        self.device = "cuda"

    def forward(self, data):
        init_x0, init_x, edge_index1, edge_index2,text = data.x0, data.x, data.edge_index, data.edge_index2, data.text
        rootindex = data.rootindex
        batch_size = max(data.batch) + 1
        root_extend = th.zeros(len(data.batch), init_x0.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = init_x0[rootindex[num_batch]]
        edge_index1 = edge_index2
        if "sibling_index" in data:
            sibling_index = data.sibling_index
        else:
            sibling_index = sibling_index=th.LongTensor([])
        x1 = self.conv1(init_x0, edge_index1,sibling_index)
        att_loss = self.conv1.get_attention_loss()
        att_sib_loss =  self.conv1.get_attention_sibling_loss()
        x1 = F.relu(x1)
        x1 = self.conv2(x1, edge_index1,sibling_index)
        att_loss += self.conv2.get_attention_loss()
        att_sib_loss += self.conv2.get_attention_sibling_loss()
        #x1 = F.relu(x1)
        if True:
            x1 = x1.view(-1,self.out_feats, 8)  # [num_nodes, out_channels,num_heads]
            x1 = self.conv3(x1)  # [num_nodes, out_channels_new,1]
            x1 = x1.squeeze(2)  # [num_nodes, out_channels_new]
        x1 = F.relu(x1)
        rootindex = data.rootindex
        batch_size = max(data.batch) + 1
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x1 = self.gate3(x1,root_extend)
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        #prod = (x1*root_extend)
        #diff = th.abs(root_extend-x1)
        att =  self.att_pool(th.cat((x1, root_extend), 0))
        att=att
        att = F.tanh(att)
        b = softmax(att, data.batch)
        if self.training:
            att =  att * (b >= 0.005).float()
        else:
            att =  att * b
        att = softmax(att, data.batch)

        new_xx = x1 * att
        x1 = scatter_add(new_xx, data.batch, dim=0)
        #x1 = scatter_mean(x1, data.batch, dim=0)
        x1_g = x1

        x = x1_g
        y = data.y1

        x = self.fc(x) 
        x = F.log_softmax(x, dim=1)
        #return x, None,None, y
        return x, att_loss,att_sib_loss, y
def preparet(fold_t):
    data= {}
    for id in fold_t:
        # Load tweets
        with open('./data/twitter16/'+ id + '/after_tweets.pkl', 'rb') as t:
            tweets = pickle.load(t)
        dict = {}
        maxindex = 0
        for index, tweet in enumerate(tweets):
            dict[tweet] = index
            if index>maxindex:
                maxindex = index
        # Load structure
        with open('./data/twitter16/'+ id + '/after_structure.pkl', 'rb') as f:
            inf = pickle.load(f)
        inf = inf[1:]
        unique_nodes = set()
        for pair in inf:
            if 'ROOT' not in pair:
                unique_nodes.update(pair)
            if len(unique_nodes) >= 5:
                break
        # Limit to the first two unique nodes
        if maxindex>=49:
            maxindex=49
        first_two_nodes = [i for i in range(maxindex+1)]
        #first_two_nodes = [i for i in range(5)]#list(unique_nodes)[:5]

        new_inf = []
        for pair in inf:
            #if pair[0] in first_two_nodes and pair[1] in first_two_nodes:
            if pair[0] != 'ROOT' and pair[1] != 'ROOT':
                if dict[pair[0]] in first_two_nodes and dict[pair[1]] in first_two_nodes:
                    new_pair = [dict[pair[0]], dict[pair[1]]]
                    new_inf.append(new_pair)
        # for pair in inf:
        #     new_pair = []
        #     for E in pair:
        #         if E == 'ROOT':
        #             break
        #         E = dict[E]
        #         new_pair.append(E)
        #     if E != 'ROOT':
        #         new_inf.append(new_pair)
        #early_rumor detection
        #new_inf = np.array(new_inf).T
        #edgeindex = new_inf
        if len(new_inf):
            new_inf = np.array(new_inf).T
            rows, cols = new_inf
            undirected_edges = [np.concatenate([rows, cols]), np.concatenate([cols, rows])]
            new_edgeindex = [list(undirected_edges[0]), list(undirected_edges[1])]
        else:
            new_edgeindex = [[], []]
        # row = list(edgeindex[0]) 
        # col = list(edgeindex[1])
        # burow = list(edgeindex[1])
        # bucol = list(edgeindex[0])
        # row.extend(burow)
        # col.extend(bucol)
        # new_edgeindex = [row, col] 

        # Load features
        with open('./bert_w2c/T16/t16_mask_015/' + id + '.json', 'r') as j_f:
        #with open('./bert_w2c/T15/twitter15_robertabase_fineacc808_cls/' + id + '.json', 'r') as j_f:
            json_inf = json.load(j_f)
        #x = json_inf[id]
        #x = np.array(x)
        # Limit features and nodes to the first two unique nodes
        #x = np.array([json_inf[id][dict[node]] for node in first_two_nodes])
        x = np.array([json_inf[id][node] for node in first_two_nodes])
        sib_edgeindex=sibling_edge(x,new_inf,0.4,False,True)

        # Load labels
        with open('./data/label_16.json', 'r') as j_tags:
            tags = json.load(j_tags)
        y = label2id[tags[id]]

        # Store in data dictionary
        data[id] = Data(x0=torch.tensor(x,dtype=torch.float32),text=[],
                                x=torch.tensor(x,dtype=torch.float32), 
                                edge_index=torch.LongTensor(new_edgeindex),
                                edge_index2=torch.LongTensor(new_edgeindex),
                                rootindex=torch.LongTensor([0]), 
                                sibling_index=torch.LongTensor(sib_edgeindex),
                                y1=torch.LongTensor([y]),
                                y2=torch.LongTensor([y]))
    return data 
def prepared(fold_x):
    tweets_dict = {}
    inf_dict = {}
    json_inf0_dict = {}
    json_inf_dict = {}
    tags_dict = {}
    edgeindex = {}
    sib_edgeindex_dict = {}
    for id in fold_x:
        # ====================================edgeindex========================================
        with open('./data/twitter16/'+ id + '/after_tweets.pkl', 'rb') as t:
            tweets = pickle.load(t)
        tweets_dict[id] = tweets

        with open('./data/twitter16/'+ id + '/after_structure.pkl', 'rb') as f:
            inf = pickle.load(f)
        inf_dict[id] = inf


        with open('./data/label_16.json', 'r') as j_tags:
            tags = json.load(j_tags)
        tags_dict[id] = label2id[tags[id]]

    # ==========================edgeindex====================================
        edgedict = {}
        maxindex = 0
        for index, tweet in enumerate(tweets):
            edgedict[tweet] = index
            if index>maxindex:
                maxindex = index

        inf = inf[1:]

        unique_nodes = set()
        for pair in inf:
            if pair[0] != 'ROOT' and pair[1] != 'ROOT':
                unique_nodes.update(pair)
            # if len(unique_nodes) >= 5:
            #     break

        unique_nodes = sorted(unique_nodes, key=lambda x: edgedict[x])#[:5]

        # Now we'll take the first two unique nodes for our graph
        if maxindex>=49:
            maxindex=49
        first_two_nodes = [i for i in range(maxindex+1)]#list(unique_nodes)#[:5]

        # Map the first two nodes to their indices
        node_indices = first_two_nodes#[edgedict[node] for node in first_two_nodes]

        # Filter the edges to include only edges between the first two nodes
        new_inf = []
        for pair in inf:
            #if pair[0] in first_two_nodes and pair[1] in first_two_nodes:
            if pair[0] != 'ROOT' and pair[1] != 'ROOT':
                if edgedict[pair[0]] in first_two_nodes and edgedict[pair[1]] in first_two_nodes:
                    new_pair = [edgedict[pair[0]], edgedict[pair[1]]]
                    new_inf.append(new_pair)
        # new_inf = []
        # for pair in inf:
        #     new_pair = []
        #     for E in pair:
        #         if E == 'ROOT':
        #             break
        #         E = edgedict[E]
        #         new_pair.append(E)
        #     if E != 'ROOT':
        #         new_inf.append(new_pair)
        
                # =========================================X===============================================
        with open('./bert_w2c/T16/t16_mask_015/' + id + '.json', 'r') as j_f0:
            json_inf0 = json.load(j_f0)
        json_inf0[id] = [json_inf0[id][index] for index in node_indices]
        json_inf0_dict[id] = json_inf0[id]
        x0 = json_inf0[id]
        x0 = np.array(x0)
        
        #print(x0.shape)
        
        with open('./bert_w2c/T16/t16_mask_015/' + id + '.json', 'r') as j_f:
            json_inf = json.load(j_f)
        json_inf[id] = [json_inf[id][index] for index in node_indices]
        json_inf_dict[id] = json_inf[id]

        edgeindex[id] = new_inf
        new_inf = np.array(new_inf).T
        sib_edgeindex_dict[id]=sibling_edge(x0,new_inf,0.4,False,True)
    dic = dict()
    dic['tweets_dict'] = tweets_dict

    dic['inf_dict'] = inf_dict
    dic['json_inf0_dict'] = json_inf0_dict
    dic['json_inf_dict'] = json_inf_dict
    dic['tags_dict'] =tags_dict
    dic['edgeindex'] =edgeindex
    dic['sib_edgeindex_dict'] = sib_edgeindex_dict
    return dic
    

def train_GCN(x_test, x_train,lr, weight_decay,patience,n_epochs,batchsize,dataname):
    model = GCN_Net(768,64,64).to(device) 
    for para in model.hard_fc1.parameters():
        para.requires_grad = False
    optimizer = th.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    

    model.train() 
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    early_stopping = EarlyStopping(patience=patience, verbose=True) 
    dic = prepared(x_train)
    dic2 = preparet(x_test)
    for epoch in range(n_epochs): 
        traindata_list, testdata_list = loadData(dataname, x_train, x_test, droprate=0.4,random=random,dic=dic,dic2=dic2) # T15 droprate = 0.1 T16 droprate = 0.4
        train_loader = DataLoader(traindata_list, batch_size=batchsize, shuffle=True, num_workers=5)
        test_loader = DataLoader(testdata_list, batch_size=batchsize, shuffle=True, num_workers=5)     
        avg_loss = [] 
        avg_acc = []
        batch_idx = 0
        #tqdm_train_loader = tqdm(train_loader)
        NUM=1
        beta=1
        for Batch_data in train_loader:
            Batch_data.to(device)
            out_labels, cl_loss,att_sib_loss, y = model(Batch_data) 
            finalloss = F.nll_loss(out_labels,y)
            loss = finalloss + beta*cl_loss + 0.01* att_sib_loss 
            avg_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, pred = out_labels.max(dim=-1)
            correct = pred.eq(y).sum().item()
            train_acc = correct / len(y) 
            avg_acc.append(train_acc)
            # print("Epoch {:05d} | Batch{:02d} | Train_Loss {:.4f}| Train_Accuracy {:.4f}".format(epoch, batch_idx,loss.item(),train_acc))
            batch_idx = batch_idx + 1
            NUM += 1
            #print('train_loss: ', loss.item())
        train_losses.append(np.mean(avg_loss))
        train_accs.append(np.mean(avg_acc))
        
        temp_val_losses = []
        temp_val_accs = []
        temp_val_Acc_all, temp_val_Acc1, temp_val_Prec1, temp_val_Recll1, temp_val_F1, \
        temp_val_Acc2, temp_val_Prec2, temp_val_Recll2, temp_val_F2, \
        temp_val_Acc3, temp_val_Prec3, temp_val_Recll3, temp_val_F3, \
        temp_val_Acc4, temp_val_Prec4, temp_val_Recll4, temp_val_F4 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        model.eval() 
        #tqdm_test_loader = tqdm(test_loader)
        for Batch_data in test_loader:
            Batch_data.to(device)
            val_out, val_re_loss,att_sib_loss, y = model(Batch_data)
            val_loss = F.nll_loss(val_out, y)
            temp_val_losses.append(val_loss.item())
            _, val_pred = val_out.max(dim=1)
            correct = val_pred.eq(y).sum().item()
            val_acc = correct / len(y) 
            Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2, Acc3, Prec3, Recll3, F3, Acc4, Prec4, Recll4, F4 = evaluation4class(
                val_pred, y) 
            temp_val_Acc_all.append(Acc_all), temp_val_Acc1.append(Acc1), temp_val_Prec1.append(
                Prec1), temp_val_Recll1.append(Recll1), temp_val_F1.append(F1), \
            temp_val_Acc2.append(Acc2), temp_val_Prec2.append(Prec2), temp_val_Recll2.append(
                Recll2), temp_val_F2.append(F2), \
            temp_val_Acc3.append(Acc3), temp_val_Prec3.append(Prec3), temp_val_Recll3.append(
                Recll3), temp_val_F3.append(F3), \
            temp_val_Acc4.append(Acc4), temp_val_Prec4.append(Prec4), temp_val_Recll4.append(
                Recll4), temp_val_F4.append(F4)
            temp_val_accs.append(val_acc)
        val_losses.append(np.mean(temp_val_losses))
        val_accs.append(np.mean(temp_val_accs))
        # print("Epoch {:05d} | Val_Loss {:.4f} | Val_Accuracy {:.4f}".format(epoch, np.mean(avg_loss), np.mean(temp_val_losses),
        #                                                                    np.mean(temp_val_accs)))
        # res = ['acc:{:.4f}'.format(np.mean(temp_val_Acc_all)),
        #        'C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc1), np.mean(temp_val_Prec1),
        #                                                np.mean(temp_val_Recll1), np.mean(temp_val_F1)),
        #        'C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc2), np.mean(temp_val_Prec2),
        #                                                np.mean(temp_val_Recll2), np.mean(temp_val_F2)),
        #        'C3:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc3), np.mean(temp_val_Prec3),
        #                                                np.mean(temp_val_Recll3), np.mean(temp_val_F3)),
        #        'C4:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc4), np.mean(temp_val_Prec4),
        #                                                np.mean(temp_val_Recll4), np.mean(temp_val_F4))]
        # print('results:', res)
      
        early_stopping(np.mean(temp_val_losses), np.mean(temp_val_accs), np.mean(temp_val_F1), np.mean(temp_val_F2),
                        np.mean(temp_val_F3), np.mean(temp_val_F4), model, 'GACL', dataname, epoch)
        accs =np.mean(temp_val_accs)
        F1 = np.mean(temp_val_F1)
        F2 = np.mean(temp_val_F2)
        F3 = np.mean(temp_val_F3)
        F4 = np.mean(temp_val_F4)
        if early_stopping.early_stop:
            print("Early stopping")
            accs=early_stopping.accs
            F1=early_stopping.F1
            F2 = early_stopping.F2
            F3 = early_stopping.F3
            F4 = early_stopping.F4
            break
    return accs,F1,F2,F3,F4


##---------------------------------main---------------------------------------
scale = 1
lr=0.0005 * scale
weight_decay=1e-4
patience=10
n_epochs=200
batchsize=120  
datasetname='Twitter16' # (1)Twitter15  (2)pheme  (3)weibo
#model="GCN" 
device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
test_accs = [] 
NR_F1 = [] # NR
FR_F1 = [] # FR
TR_F1 = [] # TR
UR_F1 = [] # UR

data_path = './data/'+datasetname.lower()+'/'
laebl_path = './data/'+datasetname+'_label_All.txt'

fold0_x_test, fold0_x_train, \
fold1_x_test,  fold1_x_train,\
fold2_x_test, fold2_x_train, \
fold3_x_test, fold3_x_train, \
fold4_x_test,fold4_x_train = load5foldData(datasetname,data_path,laebl_path,int(my_seed))

# print('fold0 shape: ', len(fold0_x_test), len(fold0_x_train))
# print('fold1 shape: ', len(fold1_x_test), len(fold1_x_train))
# print('fold2 shape: ', len(fold2_x_test), len(fold2_x_train))
# print('fold3 shape: ', len(fold3_x_test), len(fold3_x_train))
# print('fold4 shape: ', len(fold4_x_test), len(fold4_x_train))


accs0, F1_0, F2_0, F3_0, F4_0 = train_GCN(fold0_x_test,fold0_x_train,lr,weight_decay, patience,n_epochs,batchsize,datasetname)
accs1, F1_1, F2_1, F3_1, F4_1 = train_GCN(fold1_x_test,fold1_x_train,lr,weight_decay,patience,n_epochs,batchsize,datasetname)
accs2, F1_2, F2_2, F3_2, F4_2 = train_GCN(fold2_x_test,fold2_x_train,lr,weight_decay,patience,n_epochs,batchsize,datasetname)
accs3, F1_3, F2_3, F3_3, F4_3 = train_GCN(fold3_x_test,fold3_x_train,lr,weight_decay,patience,n_epochs,batchsize,datasetname)
accs4, F1_4, F2_4, F3_4, F4_4 = train_GCN(fold4_x_test,fold4_x_train,lr,weight_decay,patience,n_epochs,batchsize,datasetname)
test_accs.append((accs0+accs1+accs2+accs3+accs4)/5) 
NR_F1.append((F1_0+F1_1+F1_2+F1_3+F1_4)/5) 
FR_F1.append((F2_0 + F2_1 + F2_2 + F2_3 + F2_4) / 5) 
TR_F1.append((F3_0 + F3_1 + F3_2 + F3_3 + F3_4) / 5) 
UR_F1.append((F4_0 + F4_1 + F4_2 + F4_3 + F4_4) / 5)
print("AVG_result: {:.4f}|UR F1: {:.4f}|NR F1: {:.4f}|TR F1: {:.4f}|FR F1: {:.4f}".format(sum(test_accs), sum(NR_F1), sum(FR_F1), sum(TR_F1), sum(UR_F1)))
