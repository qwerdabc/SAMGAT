import json
import pickle
import sys,os
from numpy.matrixlib.defmatrix import matrix
sys.path.append(os.getcwd())
from Process.process_pheme import * 
#from Process.process_user import *
import torch as th
import torch
import torch.nn as nn
from torch_scatter import scatter_mean,scatter_add
from torch_geometric.utils import softmax
import torch.nn.functional as F
import numpy as np
from tools.earlystopping2class import EarlyStopping
from pytorch_lightning.callbacks import TQDMProgressBar
from torch_geometric.data import DataLoader
from tqdm import tqdm
from Process.rand5fold_pheme import *
from tools.evaluate import *
from torch_geometric.nn import GCNConv
import random
from torch_geometric.data import Data
from supergat_conv import SuperGATConv
from model.bert_cnn import BERT_CNN
from transformers import AdamW, get_linear_schedule_with_warmup
from pytorch_lightning import LightningModule
import pytorch_lightning as pl
from model.base import BaseModule
import warnings
warnings.filterwarnings("ignore")
import logging
logger = logging.getLogger(__name__)
class GatingModule(nn.Module):
    def __init__(self, input_size):
        super(GatingModule, self).__init__()
        self.W_g = nn.Linear(input_size, 1)
        self.U_g = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, hx, hc,x_local=None):
        g_cx = self.sigmoid(self.W_g(hx) + self.U_g(hc))
        refined_hx = g_cx * hx + (1 - g_cx) * hc
        #refined_hx = g_cx * hx + (1 - g_cx) * x_local
        return refined_hx


class GCN_Net(th.nn.Module): 
    def __init__(self,in_feats,hid_feats,out_feats): 
        super(GCN_Net, self).__init__() 
        self.heads = 8
        self.conv1 = SuperGATConv(in_feats, hid_feats, heads=self.heads,
                                  dropout=0.6, attention_type='MX',mode='',
                                  edge_sample_ratio=0.9, neg_sample_ratio=0.5,is_undirected=False,concat=False)
        self.conv2 = SuperGATConv(hid_feats, out_feats, heads=self.heads,
                                  dropout=0.6, attention_type='MX',mode='',
                                  edge_sample_ratio=0.9, neg_sample_ratio=0.5,is_undirected=False,concat=True)
        self.fc=th.nn.Linear(out_feats,2)
        self.conv3 = nn.Linear(self.heads, 1)
        self.out_feats = out_feats
        self.gate1 = GatingModule(in_feats)
        self.gate2 = GatingModule(hid_feats)
        self.gate3 = GatingModule(hid_feats)
        self.att_pool = nn.Linear(out_feats*1,1)
        


    def forward(self, data):
        init_x, edge_index1, edge_index2,rootindex  = data.x, data.edge_index, data.edge_index2, data.rootindex
        #print(max(edge_index1[0]),max(edge_index1[1]),len(init_x),data.id)
        batch_size = max(data.batch) + 1
        if "sibling_index" in data:
            sibling_index = data.sibling_index
        else:
            sibling_index = sibling_index=th.LongTensor([])
        root_extend = th.zeros(len(data.batch), init_x.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = init_x[rootindex[num_batch]]
        init_x = self.gate1(init_x,root_extend)
        x1 = self.conv1(init_x, edge_index1,sibling_index) # anchor
        att_loss = self.conv1.get_attention_loss()
        x1 = F.relu(x1)
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        #x1 = self.gate2(x1,root_extend)
        #x1 = F.dropout(x1, training=self.training)
        x1 = self.conv2(x1, edge_index1,sibling_index)
        att_loss += self.conv2.get_attention_loss()
        #x1 = F.relu(x1)
        if True:
            x1 = x1.view(-1,self.out_feats, self.heads)  # [num_nodes, out_channels,num_heads]
            x1 = self.conv3(x1)  # [num_nodes, out_channels_new,1]
            x1 = x1.squeeze(2)  # [num_nodes, out_channels_new]
        
        x1 = F.relu(x1)
        #print(x.shape) # 6124*64
        # pooling
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x1 = self.gate3(x1,root_extend)
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        prod = (x1*root_extend)
        diff = th.abs(root_extend-x1)
        att =  self.att_pool(diff)
        att=att#*prod.sigmoid().unsqueeze(1)
        att = F.tanh(att)
        b = softmax(att, data.batch)
        if self.training:
            att =  att * (b >= 0.005).float()
        else:
            att =  att * b
        att = softmax(att, data.batch)

        new_xx = x1 * att
        x1 = scatter_add(new_xx, data.batch, dim=0)
        #x1 = scatter_mean(x1, data.batch, dim=0) # (120, 64)
        
        x = x1
        y = data.y1

         
        x = F.log_softmax(x, dim=1) 

        return x, att_loss, y

label2id = {
            "rumor": 0,
            "non-rumor": 1,
            }
def preparet(fold_t,fold):
    data= {}
    for id in fold_t:
        # Load tweets
        with open('./data/phemeold/all/'+ id + '/tweets.pkl', 'rb') as t:
            tweets = pickle.load(t)
        dict = {}
        for index, tweet in enumerate(tweets):
            dict[tweet] = index
        
        # Load structure
        with open('./data/phemeold/all/'+ id + '/structure.pkl', 'rb') as f:
            inf = pickle.load(f)
        inf = inf[1:]
        new_inf = []
        for pair in inf:
            new_pair = []
            for E in pair:
                if E == 'ROOT':
                    break
                E = dict[E]
                new_pair.append(E)
            if E != 'ROOT':
                new_inf.append(new_pair)
        new_inf = np.array(new_inf).T
        edgeindex = new_inf
        row = list(edgeindex[0]) 
        col = list(edgeindex[1])
        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        row.extend(burow)
        col.extend(bucol)
        new_edgeindex = [row, col] 
        new_edgeindex2 = [row, col]

        if fold==1:
            with open('./data/phemeold/berttweetbasecls/' + id + '.json', 'r') as j_f:
                json_inf = json.load(j_f)
        elif fold==2:
            with open('./data/phemeold/berttweetbasecls/' + id + '.json', 'r') as j_f:
                json_inf = json.load(j_f)
        elif fold==3:
            with open('./data/phemeold/berttweetbasecls/' + id + '.json', 'r') as j_f:
                json_inf = json.load(j_f)
        elif fold==4:
            with open('./data/phemeold/berttweetbasecls/' + id + '.json', 'r') as j_f:
                json_inf = json.load(j_f)
        elif fold==5:
            with open('./data/phemeold/berttweetbasecls/' + id + '.json', 'r') as j_f:
                json_inf = json.load(j_f)
        x = json_inf[id]
        x = np.array(x)

        # Load labels
        with open('./data/phemeold/pheme_label.json', 'r') as j_tags:
            tags = json.load(j_tags)
        y = label2id[tags[id]]

        # Store in data dictionary
        data[id] = Data(x0=torch.tensor(x,dtype=torch.float32),
                                x=torch.tensor(x,dtype=torch.float32), 
                                edge_index=torch.LongTensor(new_edgeindex),
                                edge_index2=torch.LongTensor(new_edgeindex2),
                                rootindex=torch.LongTensor([0]), 
                                y1=torch.LongTensor([y]),
                                y2=torch.LongTensor([y]))
    return data 
def prepared(fold_x,fold):
    tweets_dict = {}
    inf_dict = {}
    json_inf0_dict = {}
    json_inf_dict = {}
    tags_dict = {}
    edgeindex = {}
    texts= {}
    for id in fold_x:
        with open('./data/phemeold/all/'+ id + '/source-tweet/' + id+'.json', 'rb') as t:
            text = json.load(t)
            text = text['text']
        texts[id] = text
        # ====================================edgeindex========================================
        with open('./data/phemeold/all/'+ id + '/tweets.pkl', 'rb') as t:
            tweets = pickle.load(t)
        tweets_dict[id] = tweets

        with open('./data/phemeold/all/'+ id + '/structure.pkl', 'rb') as f:
            inf = pickle.load(f)
        inf_dict[id] = inf

        # =========================================X===============================================
        if fold==1:
            with open('./data/phemeold/berttweetbasecls/' + id + '.json', 'r') as j_f:
                json_inf = json.load(j_f)
        elif fold==2:
            with open('./data/phemeold/berttweetbasecls/' + id + '.json', 'r') as j_f:
                json_inf = json.load(j_f)
        elif fold==3:
            with open('./data/phemeold/berttweetbasecls/' + id + '.json', 'r') as j_f:
                json_inf = json.load(j_f)
        elif fold==4:
            with open('./data/phemeold/berttweetbasecls/' + id + '.json', 'r') as j_f:
                json_inf = json.load(j_f)
        elif fold==5:
            with open('./data/phemeold/berttweetbasecls/' + id + '.json', 'r') as j_f:
                json_inf = json.load(j_f)
        json_inf_dict[id] = json_inf[id]

        with open('./data/phemeold/pheme_label.json', 'r') as j_tags:
            tags = json.load(j_tags)
        tags_dict[id] = label2id[tags[id]]

    # ==========================edgeindex====================================
        edgedict = {}
        for index, tweet in enumerate(tweets):
            edgedict[tweet] = index

        inf = inf[1:]
        new_inf = []
        for pair in inf:
            new_pair = []
            for E in pair:
                if E == 'ROOT':
                    break
                E = edgedict[E]
                new_pair.append(E)
            if E != 'ROOT':
                new_inf.append(new_pair)
        new_inf = np.array(new_inf).T
        edgeindex[id] = new_inf
    dic = dict()
    dic['tweets_dict'] = tweets_dict

    dic['inf_dict'] = inf_dict
    dic['json_inf0_dict'] = json_inf0_dict
    dic['json_inf_dict'] = json_inf_dict
    dic['tags_dict'] =tags_dict
    dic['edgeindex'] =edgeindex
    dic['text'] = texts
    return dic
    
def preparebert(fold_b):
    texts= {}
    tagss = {}
    for id in fold_b:
        with open('./data/phemeold/berttoken/'+ id +'.json', 'rb') as t:
            text = json.load(t)
            text = text[id][0]
        texts[id] = text
        with open('./data/phemeold/pheme_label.json', 'r') as j_tags:
            tags = json.load(j_tags)
        tagss[id] = tags
    dic= {}
    dic['texts'] = texts
    dic['tags'] = tagss
    return dic
    
def train_GCN(x_test, x_train,lr, weight_decay,patience,n_epochs,batchsize,dataname,fold): 
    model = GCN_Net(768,64,64).to(device) 
    optimizer = th.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)

    model.train() 
    train_losses = [] 
    val_losses = [] 
    train_accs = [] 
    val_accs = [] 
    early_stopping = EarlyStopping(patience=patience, verbose=True) 

    dic = prepared(x_train,fold)
    dic2 = preparet(x_test,fold)
    for epoch in range(n_epochs): 
        traindata_list, testdata_list = loadUdData(dataname, x_train, x_test, droprate=0,dic=dic,dic2=dic2)#0.3) 
        train_loader = DataLoader(traindata_list, batch_size=batchsize, shuffle=True, num_workers=10)
        test_loader = DataLoader(testdata_list, batch_size=batchsize, shuffle=True, num_workers=10)       
        avg_loss = [] 
        avg_acc = []
        batch_idx = 0
        model.train() 
        #tqdm_train_loader = tqdm(train_loader)
        NUM=1
        for Batch_data in train_loader:
            Batch_data.to(device)
            out_labels, cl_loss, y = model(Batch_data)
            finalloss=F.nll_loss(out_labels,y) 
            loss = finalloss + 1*cl_loss
            
            optimizer.zero_grad()
            loss.backward()
            avg_loss.append(loss.item()) 
            optimizer.step()

            _, pred = out_labels.max(dim=-1)
            correct = pred.eq(y).sum().item()
            train_acc = correct / len(y) 
            avg_acc.append(train_acc)
            # print("Iter {:03d} | Epoch {:05d} | Batch{:02d} | Train_Loss {:.4f}| Train_Accuracy {:.4f}".format(iter,epoch, batch_idx,
            #                                                                                      loss.item(),
            #                                                                                      train_acc))
            batch_idx = batch_idx + 1
            NUM += 1

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
            val_out, val_re_loss, y = model(Batch_data)
            val_loss = F.nll_loss(val_out, y)
            temp_val_losses.append(val_loss.item())
            _, val_pred = val_out.max(dim=1) 
            correct = val_pred.eq(y).sum().item()
            val_acc = correct / len(y) 
            Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2 = evaluationclass(
                val_pred, y)
            temp_val_Acc_all.append(Acc_all), temp_val_Acc1.append(Acc1), temp_val_Prec1.append(
                Prec1), temp_val_Recll1.append(Recll1), temp_val_F1.append(F1), \
            temp_val_Acc2.append(Acc2), temp_val_Prec2.append(Prec2), temp_val_Recll2.append(
                Recll2), temp_val_F2.append(F2)
            temp_val_accs.append(val_acc)
        val_losses.append(np.mean(temp_val_losses))
        val_accs.append(np.mean(temp_val_accs))
        # print("Epoch {:05d} | Val_Loss {:.4f}| Val_Accuracy {:.4f}".format(epoch, np.mean(temp_val_losses),
        #                                                                    np.mean(temp_val_accs)))

        res = ['acc:{:.4f}'.format(np.mean(temp_val_Acc_all)),
               'C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc1), np.mean(temp_val_Prec1),
                                                       np.mean(temp_val_Recll1), np.mean(temp_val_F1)),
               'C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc2), np.mean(temp_val_Prec2),
                                                       np.mean(temp_val_Recll2), np.mean(temp_val_F2))]
        #print('results:', res)
        early_stopping(np.mean(temp_val_losses), np.mean(temp_val_Acc_all), np.mean(temp_val_Acc1),
                       np.mean(temp_val_Acc2), np.mean(temp_val_Prec1),
                       np.mean(temp_val_Prec2), np.mean(temp_val_Recll1), np.mean(temp_val_Recll2),
                       np.mean(temp_val_F1),
                       np.mean(temp_val_F2), model, 'BiGCN', "weibo")
        accs = np.mean(temp_val_Acc_all)
        acc1 = np.mean(temp_val_Acc1)
        acc2 = np.mean(temp_val_Acc2)
        pre1 = np.mean(temp_val_Prec1)
        pre2 = np.mean(temp_val_Prec2)
        rec1 = np.mean(temp_val_Recll1)
        rec2 = np.mean(temp_val_Recll2)
        F1 = np.mean(temp_val_F1)
        F2 = np.mean(temp_val_F2)
        if early_stopping.early_stop:
            print("Early stopping")
            accs = early_stopping.accs
            acc1 = early_stopping.acc1
            acc2 = early_stopping.acc2
            pre1 = early_stopping.pre1
            pre2 = early_stopping.pre2
            rec1 = early_stopping.rec1
            rec2 = early_stopping.rec2
            F1 = early_stopping.F1
            F2 = early_stopping.F2
            break
    return train_losses, val_losses, train_accs, val_accs, accs, acc1, pre1, rec1, F1, acc2, pre2, rec2, F2



##---------------------------------main---------------------------------------


def setup_seed(seed):
     th.manual_seed(seed) 
     th.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     th.backends.cudnn.deterministic = True


setup_seed(int(30))


lr=0.0005 
weight_decay=1e-4 
patience=15  
n_epochs=200 
batchsize=120 # twitter
#batchsize=16 # weibo
datasetname='Phemeold'
iterations=1
model="GCN"
device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')


fold0_x_test, fold0_x_train,\
    fold1_x_test, fold1_x_train, \
    fold2_x_test, fold2_x_train,  \
    fold3_x_test, fold3_x_train,  \
    fold4_x_test, fold4_x_train = load5foldData(datasetname,25)
# with open('split.pkl', 'wb') as f:
#     pickle.dump((fold0_x_test, fold0_x_train,
#                  fold1_x_test, fold1_x_train,
#                  fold2_x_test, fold2_x_train,
#                  fold3_x_test, fold3_x_train,
#                  fold4_x_test, fold4_x_train), f)

# print('fold0 shape: ', len(fold0_x_test), len(fold0_x_train))
# print('fold1 shape: ', len(fold1_x_test), len(fold1_x_train))
# print('fold2 shape: ', len(fold2_x_test), len(fold2_x_train))
# print('fold3 shape: ', len(fold3_x_test), len(fold3_x_train))
# print('fold4 shape: ', len(fold4_x_test), len(fold4_x_train))


test_accs,ACC1,ACC2,PRE1,PRE2,REC1,REC2,F1,F2 = [],[],[],[],[],[],[],[],[]

for iter in range(iterations):
    train_losses, val_losses, train_accs, val_accs, accs_0, acc1_0, pre1_0, rec1_0, F1_0, acc2_0, pre2_0, rec2_0, F2_0 = train_GCN(
                                                                                               fold0_x_test,
                                                                                               fold0_x_train,
                                                                                        
                                                                                               lr, weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               1)
    train_losses, val_losses, train_accs, val_accs,accs_1, acc1_1, pre1_1, rec1_1, F1_1, acc2_1, pre2_1, rec2_1, F2_1 = train_GCN(
                                                                                               fold1_x_test,
                                                                                               fold1_x_train,
                                                                                               lr,
                                                                                               weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               2)
    train_losses, val_losses, train_accs, val_accs, accs_2, acc1_2, pre1_2, rec1_2, F1_2, acc2_2, pre2_2, rec2_2, F2_2 = train_GCN(
                                                                                               fold2_x_test,
                                                                                               fold2_x_train,
                                                                                               lr,
                                                                                               weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               3)
    train_losses, val_losses, train_accs, val_accs, accs_3, acc1_3, pre1_3, rec1_3, F1_3, acc2_3, pre2_3, rec2_3, F2_3 = train_GCN(
                                                                                               fold3_x_test,
                                                                                               fold3_x_train,
                                                                                               lr,
                                                                                               weight_decay,
                                                                                               patience,
                                                                                                   n_epochs,
                                                                                                   batchsize,
                                                                                                   datasetname,
                                                                                                   4)
    train_losses, val_losses, train_accs, val_accs, accs_4, acc1_4, pre1_4, rec1_4, F1_4, acc2_4, pre2_4, rec2_4, F2_4 = train_GCN(
                                                                                               fold4_x_test,
                                                                                               fold4_x_train,
                                                                                                lr,
                                                                                               weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               5)
    test_accs.append((accs_0+accs_1+accs_2+accs_3+accs_4)/5)
    ACC1.append((acc1_0 + acc1_1 + acc1_2 + acc1_3 + acc1_4) / 5)
    ACC2.append((acc2_0 + acc2_1 +acc2_2 + acc2_3 +acc2_4) / 5)
    PRE1.append((pre1_0 + pre1_1 + pre1_2 + pre1_3 + pre1_4) / 5)
    PRE2.append((pre2_0 + pre2_1 + pre2_2 + pre2_3 + pre2_4) / 5)
    REC1.append((rec1_0 + rec1_1 + rec1_2 + rec1_3 + rec1_4) / 5)
    REC2.append((rec2_0 + rec2_1 + rec2_2 + rec2_3 + rec2_4) / 5)
    F1.append((F1_0+F1_1+F1_2+F1_3+F1_4)/5)
    F2.append((F2_0 + F2_1 + F2_2 + F2_3 + F2_4) / 5)
print("pheme:|Total_Test_ Accuracy: {:.4f}|acc1: {:.4f}|acc2: {:.4f}|pre1: {:.4f}|pre2: {:.4f}"
                  "|rec1: {:.4f}|rec2: {:.4f}|F1: {:.4f}|F2: {:.4f}".format(sum(test_accs) / iterations, sum(ACC1) / iterations,
                                                                            sum(ACC2) / iterations, sum(PRE1) / iterations, sum(PRE2) /iterations,
                                                                            sum(REC1) / iterations, sum(REC2) / iterations, sum(F1) / iterations, sum(F2) / iterations))

