from collections import defaultdict, deque
import os
import numpy as np
import torch
#import random
from torch.utils.data import Dataset
from torch_geometric.data import Data
import pickle
#from transformers import *
import json
from torch.utils.data import DataLoader


# global
label2id = {
            "unverified": 0,
            "non-rumor": 1,
            "true": 2,
            "false": 3,
            }

def random_pick(list, probabilities,random): 
    x = random.uniform(0,1)
    cumulative_probability = 0.0 
    for item, item_probability in zip(list, probabilities): 
         cumulative_probability += item_probability 
         if x < cumulative_probability:
               break 
    return item 

def similarity_select(x,new_edge_index,percentage):
    x = torch.tensor(x,dtype=torch.float32)
    normalized_x = x / x.norm(dim=-1, keepdim=True)

    # 计算 new_edge_index 对应的边的特征相似度
    similarity_scores = [torch.dot(normalized_x[u], normalized_x[v]).item() for u, v in new_edge_index.transpose().tolist()]
    #print(similarity_scores)
    # 对相似度进行排序并选取前百分比的边
    num_top_edges = max(1, int(len(similarity_scores) * percentage))  # 保证至少选择一个边
    sorted_indices = sorted(range(len(similarity_scores)), key=lambda i: similarity_scores[i], reverse=True)[:num_top_edges]
    # 提取 ratio_edge_index
    ratio_edge_index = new_edge_index.T[sorted_indices].T
    #print(ratio_edge_index)
    return ratio_edge_index
def similarity_select_n(x, new_edge_index, n, threshold=0.6, k=4):
    x = torch.tensor(x, dtype=torch.float32)
    normalized_x = x / x.norm(dim=-1, keepdim=True)
    #print("similarity_index",new_edge_index)
    # Calculate the similarity scores for the edges in new_edge_index
    similarity_scores = [torch.dot(normalized_x[u], normalized_x[v]).item() for u, v in new_edge_index.transpose().tolist()]

    # Sort the scores and indices 
    sorted_indices_scores = sorted(enumerate(similarity_scores), key=lambda x: x[1], reverse=True)

    # Keep track of how many edges have been added for each node
    node_edge_counts = defaultdict(int)
    
    # Keep track of the degree of each node
    node_degrees = defaultdict(int)

    # The indices of the edges to keep
    keep_indices = []

    # Initialize node degrees
    for u, v in new_edge_index.transpose().tolist():
        node_degrees[u] += 1
        node_degrees[v] += 1
    
    # Loop over the sorted (index, score) tuples
    for index, score in sorted_indices_scores:
        # Only consider edges with similarity score above the threshold
        if score < threshold:
            continue

        # Get the nodes this edge is connected to
        u, v = new_edge_index.transpose()[index]
        
        # If adding this edge does not cause either node to exceed the limit
        if node_edge_counts[u] < n and node_edge_counts[v] < n:
            # Check if adding this edge would cause the degree of either node to exceed k
            if node_degrees[u] < k and node_degrees[v] < k:
                # Add this edge
                keep_indices.append(index)
                
                # Update the edge counts and degrees
                node_edge_counts[u] += 1
                node_edge_counts[v] += 1
                node_degrees[u] += 1
                node_degrees[v] += 1
            
    # Extract the edges to keep
    ratio_edge_index = new_edge_index.T[keep_indices].T
    return ratio_edge_index

def sibling_edge_parent(x,edge_index,ratio,sta=True,limitn=False,z=3):
    def bfs(edge_index, root):
        visited = set([root])
        layers = {0: [root]}
        parent_map = {root: None}
        queue = deque([(root, 0)])

        while queue:
            node, level = queue.popleft()
            next_level = level + 1
            if next_level not in layers:
                layers[next_level] = []

            for i in range(edge_index.shape[1]):
                if edge_index[0, i] == node:
                    neighbor = edge_index[1, i]
                    if neighbor not in visited:
                        visited.add(neighbor)
                        layers[next_level].append(neighbor)
                        parent_map[neighbor] = node
                        queue.append((neighbor, next_level))

        return layers, parent_map
    root = [0]#set(edge_index[0]) - set(edge_index[1])
    if len(root) != 1:
        raise ValueError("more than 1 root")
    root = root.pop()
    layers, parent_map = bfs(edge_index, root)

    # Group sibling nodes by their parent
    siblings_by_parent = defaultdict(list)
    for layer_nodes in layers.values():
        for i in range(len(layer_nodes)):
            for j in range(i + 1, len(layer_nodes)):
                if parent_map[layer_nodes[i]] == parent_map[layer_nodes[j]]:
                    parent = parent_map[layer_nodes[i]]
                    siblings_by_parent[parent].append([layer_nodes[i], layer_nodes[j]])
    # Select top z sibling edges for each parent
    new_edges = []
    for parent, sibling_edges in siblings_by_parent.items():
        if len(sibling_edges) > z:
            sibling_edges = similarity_select_n(x, np.array(sibling_edges).T, 1)  # Assumes x is not a torch tensor here
            sibling_edges = sibling_edges.T.tolist()[:z]
        new_edges.extend(sibling_edges)
    #print(edge_index,new_edges)
    new_edge_index = np.array(new_edges).T
    if len(new_edge_index) == 0:
        if sta is True:
            updated_edge_index = edge_index
        else:
            updated_edge_index = []
    else:
        #new_edge_index = similarity_select(x,new_edge_index,ratio)
        #new_edge_index = random_select(new_edge_index,ratio)
        if sta is True:
            updated_edge_index = np.hstack([edge_index,new_edge_index])
        else:
            updated_edge_index = new_edge_index
    return updated_edge_index


def sibling_edge(x,edge_index,ratio,sta=True,limitn=False):
    if len(edge_index) == 0:
        return []
    #print("edge_index",edge_index)
    def bfs(edge_index, root):
        visited = set([root])
        layers = {0: [root]}
        parent_map = {root: None}
        queue = deque([(root, 0)])

        while queue:
            node, level = queue.popleft()
            next_level = level + 1
            if next_level not in layers:
                layers[next_level] = []

            for i in range(edge_index.shape[1]):
                if edge_index[0, i] == node:
                    neighbor = edge_index[1, i]
                    if neighbor not in visited:
                        visited.add(neighbor)
                        layers[next_level].append(neighbor)
                        parent_map[neighbor] = node
                        queue.append((neighbor, next_level))

        return layers, parent_map
    root = [0]#set(edge_index[0]) - set(edge_index[1])
    if len(root) != 1:
        raise ValueError("more than 1 root")
    root = root.pop()
    layers, parent_map = bfs(edge_index, root)

    # 添加具有相同父节点的兄弟节点之间的边（无向图）
    new_edges = []
    for layer_nodes in layers.values():
        for i in range(len(layer_nodes)):
            for j in range(i + 1, len(layer_nodes)):
                if parent_map[layer_nodes[i]] == parent_map[layer_nodes[j]]:
                    new_edges.append([layer_nodes[i], layer_nodes[j]])
                    new_edges.append([layer_nodes[j], layer_nodes[i]])
    #print(edge_index,new_edges)
    new_edge_index = np.array(new_edges).T
    if len(new_edge_index) == 0:
        if sta is True:
            updated_edge_index = edge_index
        else:
            updated_edge_index = []
    else:
        if limitn is False:
            new_edge_index = similarity_select(x,new_edge_index,ratio)
        else:
            new_edge_index = similarity_select_n(x,new_edge_index,1)
        #new_edge_index = similarity_select(x,new_edge_index,ratio)
        #new_edge_index = random_select(new_edge_index,ratio)
        if sta is True:
            updated_edge_index = np.hstack([edge_index,new_edge_index])
        else:
            updated_edge_index = new_edge_index
    return updated_edge_index



class RumorDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        #item = {key: torch.LongTensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])#torch.LongTensor(batch_label).to(device1)
        #item['labels'] = torch.LongTensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class GraphDataset(Dataset):
    def __init__(self, fold_x, droprate,random,dic): 
        self.random = random
        self.fold_x = fold_x
        self.droprate = droprate
        self.tweets_dict = dic['tweets_dict'] # {}
        self.inf_dict = dic['inf_dict']#{}
        self.json_inf0_dict = dic['json_inf0_dict']#{}
        self.json_inf_dict = dic['json_inf_dict']#{}
        self.tags_dict = dic['tags_dict']#{}
        self.edgeindex = dic['edgeindex']
        self.sib_edgeindex_dict = dic['sib_edgeindex_dict']
        # self.tweets_dict = {}
        # self.inf_dict = {}
        # self.json_inf0_dict = {}
        # self.json_inf_dict = {}
        # self.tags_dict = {}
        self.dataname = "twitter16"
        self.short = "16"


    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index): 
        id =self.fold_x[index]
    
        # ====================================edgeindex========================================
        tweets = self.tweets_dict[id]
        #print(tweets)
        dict = {}
        for index, tweet in enumerate(tweets):
            dict[tweet] = index
        #print('dict: ', dict)

        # inf = self.inf_dict[id]

        # inf = inf[1:]
        # new_inf = []
        # for pair in inf:
        #     new_pair = []
        #     for E in pair:
        #         if E == 'ROOT':
        #             break
        #         E = dict[E]
        #         new_pair.append(E)
        #     if E != 'ROOT':
        #         new_inf.append(new_pair)
        # new_inf = np.array(new_inf).T
        # edgeindex = new_inf
        edgeindex = self.edgeindex[id]
        
        if len(edgeindex):
            edgeindex = np.array(edgeindex).T
            init_row = list(edgeindex[0]) 
            init_col = list(edgeindex[1]) 
            rows, cols = edgeindex
            undirected_edges = [np.concatenate([rows, cols]), np.concatenate([cols, rows])]
            new_edgeindex = [list(undirected_edges[0]), list(undirected_edges[1])]
            row,col = new_edgeindex
        else:
            new_edgeindex = [[], []]
            init_row = []
            init_col = []
            row,col = [],[]
        # init_row = list(edgeindex[0]) 
        # init_col = list(edgeindex[1]) 
        # burow = list(edgeindex[1]) 
        # bucol = list(edgeindex[0]) 
        # row = init_row + burow 
        # col = init_col + bucol
      
        # new_edgeindex = [row, col]
        

        #==================================- dropping + adding + misplacing -===================================#

        choose_list = [1,2,3] # 1-drop 2-add 3-misplace
        probabilities = [0.7,0.2,0.1] # T15: probabilities = [0.5,0.3,0.2] 
        choose_num = random_pick(choose_list, probabilities,self.random)

        if self.droprate > 0:
            if choose_num == 1:
            
                length = len(row)
                poslist = self.random.sample(range(length), int(length * (1 - self.droprate)))
                poslist = sorted(poslist)
                row2 = list(np.array(row)[poslist])
                col2 = list(np.array(col)[poslist])
                new_edgeindex2 = [row2, col2]
                #new_edgeindex = [row2, col2]
                '''
                length = len(list(set(sorted(row))))
                print('length:', length)
                poslist = self.random.sample(range(1,length), int(length * self.prerate))
                print('len of poslist: ', len(poslist))
                new_row = []
                new_col = []
                #print('row:',row)
                #print('poslist', poslist)
                for i_r, e_r in enumerate(row):
                    for i_c, e_c in enumerate(col):
                        if i_r == i_c:
                            if e_r not in poslist and e_c not in poslist:
                                new_row.append(e_r)
                                new_col.append(e_c)
                                #print('new_row:', new_row)
                                #print('new_col:', new_col)
                    
                print('len of new_row:', len(new_row))
                if len(new_row) != len(new_col):
                    print('setting error')
                Dict = {}
                for index, tweet in enumerate(sorted(list(set(new_row+new_col)))):
                    Dict[tweet] = index
                
                row2 = []
                col2 = []
                for i_nr in new_row:
                    row2.append(Dict[i_nr])
                for i_nc in new_col:
                    col2.append(Dict[i_nc])
                #print('row2:',row2)
                '''
                
                
            elif choose_num == 2:
                '''
                length = len(row)
                last_num = list(set(sorted(row)))[-1]
                add_list = list(range(last_num+1, int(length * self.prerate)))
                add_row = []
                add_col = []
                for add_item in add_list:
                    add_row.append(add_item)
                    add_col.append(self.random.randint(0, add_item-1))
                row2 = row + add_row + add_col
                col2 = col + add_col + add_row
                '''
                length = len(list(set(sorted(row))))
                add_row = self.random.sample(range(length), int(length * self.droprate)) 
                add_col = self.random.sample(range(length), int(length * self.droprate))
                row2 = row + add_row + add_col
                col2 = col + add_col + add_row

                new_edgeindex2 = [row2, col2]


                         
            elif choose_num == 3: 
                length = len(init_row)
                mis_index_list = self.random.sample(range(length), int(length * self.droprate))
                #print('mis_index_list:', mis_index_list)
                Sort_len = len(list(set(sorted(row))))
                if Sort_len > int(length * self.droprate):
                    mis_value_list = self.random.sample(range(Sort_len), int(length * self.droprate))
                    #print('mis_valu_list:', mis_value_list)
                    #val_i = 0
                    for i, item in enumerate(init_row):
                        for mis_i,mis_item in enumerate(mis_index_list):
                            if i == mis_item and mis_value_list[mis_i] != item:
                                init_row[i] = mis_value_list[mis_i]
                    row2 = init_row + init_col
                    col2 = init_col + init_row
                    new_edgeindex2 = [row2, col2]


                else:
                    length = len(row)
                    poslist = self.random.sample(range(length), int(length * (1 - self.droprate)))
                    poslist = sorted(poslist)
                    row2 = list(np.array(row)[poslist])
                    col2 = list(np.array(col)[poslist])
                    new_edgeindex2 = [row2, col2]
        else:
             new_edgeindex = [row, col]
             new_edgeindex2 = [row, col]
        
        
        json_inf0 = self.json_inf0_dict#[id]
        
        x0 = json_inf0[id]
        x0 = np.array(x0)
        
        json_inf = self.json_inf_dict#[id]
        
        x_list = json_inf[id]
        x = np.array(x_list)
        



        y = self.tags_dict[id]#label2id[tags[id]]
        #y = np.array(y)
        if self.droprate > 0:
            if choose_num == 1:
                zero_list = [0]*768
                x_length = len(x_list)
                r_list = self.random.sample(range(x_length), int(x_length * self.droprate))
                r_list = sorted(r_list)
                for idex, line in enumerate(x_list):
                    for r in r_list:
                        if idex == r:
                            x_list[idex] = zero_list
                
                x2 = np.array(x_list)
                x = x2
        #sib_edgeindex=sibling_edge(x0,edgeindex,0.4,False,True)
        sib_edgeindex= self.sib_edgeindex_dict[id]

        return Data(text = [],#text,
                    sibling_index=torch.LongTensor(sib_edgeindex),
                    x0=torch.tensor(x0,dtype=torch.float32),
                    x=torch.tensor(x,dtype=torch.float32), 
                    edge_index=torch.LongTensor(new_edgeindex),
                    edge_index2=torch.LongTensor(new_edgeindex2),
                    rootindex=torch.LongTensor([0]),
                    y1=torch.LongTensor([y]),
                    y2=torch.LongTensor([y])) 



class test_GraphDataset(Dataset):
    def __init__(self, fold_x, droprate,dic): 
        
        self.fold_x = fold_x
        self.droprate = droprate
        self.dataname = "twitter16"
        self.short = "16"
        self.data = dic

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index): 
        id =self.fold_x[index] 
        return self.data[id]
        # ====================================edgeindex==============================================
        with open('./data/'+self.dataname+'/'+ id + '/after_tweets.pkl', 'rb') as t:
            tweets = pickle.load(t)
        #print(tweets)
        dict = {}
        for index, tweet in enumerate(tweets):
            dict[tweet] = index
        #print('dict: ', dict)

        with open('./data/'+self.dataname+'/'+ id + '/after_structure.pkl', 'rb') as f:
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

 =========================================X====================================================
        with open('./bert_w2c/T15/t15_mask_00/' + id + '.json', 'r') as j_f:
            json_inf = json.load(j_f)
        
        x = json_inf[id]
        x = np.array(x)

        with open('./data/label_'+self.short+'.json', 'r') as j_tags:
            tags = json.load(j_tags)

        y = label2id[tags[id]]
        #y = np.array(y)


        return Data(text=[],#text, 
                    x0=torch.tensor(x,dtype=torch.float32),
                    x=torch.tensor(x,dtype=torch.float32), 
                    edge_index=torch.LongTensor(new_edgeindex),
                    edge_index2=torch.LongTensor(new_edgeindex2),
                    rootindex=torch.LongTensor([0]), 
                    y1=torch.LongTensor([y]),
                    y2=torch.LongTensor([y])) 
