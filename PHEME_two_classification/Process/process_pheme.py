import os
from Process.dataset_pheme import UdGraphDataset, test_UdGraphDataset,UdGraphDatasetbert

cwd=os.getcwd()

def loadBertData(dataname, fold_x_train,fold_x_test,dic,dic2):
    
    #print("loading train set", )
    traindata_list = UdGraphDatasetbert(fold_x_train, dic=dic)
    #print("train no:", len(traindata_list))
    #print("loading test set", )
    testdata_list = UdGraphDatasetbert(fold_x_test, dic=dic2) # droprate*****
    #print("test no:", len(testdata_list))
    return traindata_list, testdata_list

def loadUdData(dataname, fold_x_train,fold_x_test,droprate,dic,dic2):
    
    #print("loading train set", )
    traindata_list = UdGraphDataset(fold_x_train, droprate=droprate,dic=dic)
    #print("train no:", len(traindata_list))
    #print("loading test set", )
    testdata_list = test_UdGraphDataset(fold_x_test, droprate=0,dic=dic2) # droprate*****
    #print("test no:", len(testdata_list))
    return traindata_list, testdata_list
