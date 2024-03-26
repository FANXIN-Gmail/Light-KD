# -- coding:UTF-8
import numpy as np 
import pandas as pd 
import scipy.sparse as sp 
import scipy

import torch.nn as nn 
import torch.utils.data as data
import pdb
from torch.autograd import Variable
from torch import linalg as LA
import torch
import math
import random
import collections
import json

from sklearn.cluster import KMeans
import torch.nn.functional as F

t = 1
n_clusters = 5
n_preferences = 32

class BPR(nn.Module):
    def __init__(self, user_num,item_num,factor_num,user_item_matrix,item_user_matrix,
        local_u_i=None,
        local_i_u=None,
        old_U_emb=None,
        old_I_emb=None,
        old_User=None,
        u_sample=None,
        i_sample=None):
        super(BPR, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors.
        """

        # Original LightGCN
        self.user_item_matrix = user_item_matrix
        self.item_user_matrix = item_user_matrix
        self.user_num = user_num
        self.item_num = item_num
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)
        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

        if old_U_emb == None and old_I_emb == None:
            pass
        else:
            # Distillation 
            self.local_u_i = local_u_i
            self.local_i_u = local_i_u
            self.old_U_emb = old_U_emb
            self.old_I_emb = old_I_emb

            self.old_User = old_User

            self.u_sample = u_sample
            self.i_sample = i_sample

    def extra_define(self, factor_num):

        self.preference = nn.Sequential(
            nn.Linear(factor_num, factor_num),
            nn.LeakyReLU(),
            nn.Linear(factor_num, factor_num),
            nn.LeakyReLU(),
            nn.Linear(factor_num, n_preferences)
            )

        self.shift = nn.Sequential(
            nn.Linear(n_preferences, factor_num),
            nn.ReLU(),
            nn.Linear(factor_num, 1),
            nn.Softplus()
            )

        for layer in self.preference:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.01)
        for layer in self.shift:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.01)

    def loss_BPR(self, user, item_i, item_j, gcn_users_embedding, gcn_items_embedding):

        user = F.embedding(user,gcn_users_embedding)
        item_i = F.embedding(item_i,gcn_items_embedding)
        item_j = F.embedding(item_j,gcn_items_embedding)

        prediction_i = (user * item_i).sum(dim=-1)
        prediction_j = (user * item_j).sum(dim=-1)

        BPR = -((prediction_i - prediction_j).sigmoid().log().mean())
        regulization = 0.0001*(user**2+item_i**2+item_j**2).sum(dim=-1).mean()

        return BPR + regulization

    def PIW(self, new_User):

        p_new = self.preference(new_User[1])
        p_old = self.preference(self.old_User[1])
        p_new, p_old = nn.functional.softmax(p_new, dim=1), nn.functional.softmax(p_old, dim=1)
        s_u = (p_old - p_new) * (p_old - p_new)
        w_u = self.shift(s_u)

        return w_u

    def loss_KD(self, gcn_users_embedding, gcn_items_embedding, w_u):

        w_u = F.embedding(self.u_sample, w_u)

        print(w_u[0:10].t())

        user_new = F.embedding(self.u_sample, gcn_users_embedding)
        item_i_new = F.embedding(self.i_sample, gcn_items_embedding)

        user_old = F.embedding(self.u_sample, self.old_U_emb)
        item_i_old = F.embedding(self.i_sample, self.old_I_emb)

        Positive = (w_u*( (torch.mul(user_new, item_i_new).sum(dim=-1) - torch.mul(user_old, item_i_old).sum(dim=-1))**2 )).mean()

        return Positive

    def forward(self, user, item_i, item_j):

        users_embedding=self.embed_user.weight
        items_embedding=self.embed_item.weight

        gcn1_users_embedding = torch.sparse.mm(self.user_item_matrix, items_embedding) #+ users_embedding.mul(self.d_i_train))#*2. #+ users_embedding
        gcn1_items_embedding = torch.sparse.mm(self.item_user_matrix, users_embedding) #+ items_embedding.mul(self.d_j_train))#*2. #+ items_embedding

        gcn2_users_embedding = torch.sparse.mm(self.user_item_matrix, gcn1_items_embedding) #+ gcn1_users_embedding.mul(self.d_i_train))#*2. + users_embedding
        gcn2_items_embedding = torch.sparse.mm(self.item_user_matrix, gcn1_users_embedding) #+ gcn1_items_embedding.mul(self.d_j_train))#*2. + items_embedding

        gcn3_users_embedding = torch.sparse.mm(self.user_item_matrix, gcn2_items_embedding) #+ gcn2_users_embedding.mul(self.d_i_train))#*2. + gcn1_users_embedding
        gcn3_items_embedding = torch.sparse.mm(self.item_user_matrix, gcn2_users_embedding) #+ gcn2_items_embedding.mul(self.d_j_train))#*2. + gcn1_items_embedding

        gcn_users_embedding = users_embedding + (1/2)*gcn1_users_embedding + (1/3)*gcn2_users_embedding + (1/4)*gcn3_users_embedding
        gcn_items_embedding = items_embedding + (1/2)*gcn1_items_embedding + (1/3)*gcn2_items_embedding + (1/4)*gcn3_items_embedding

        w_u = self.PIW([users_embedding, gcn1_users_embedding])

        loss_BPR = self.loss_BPR(user, item_i, item_j, gcn_users_embedding, gcn_items_embedding)
        loss_KD = self.loss_KD(gcn_users_embedding, gcn_items_embedding, w_u)

        return loss_BPR + 0.01*loss_KD

    def inference(self):

        users_embedding=self.embed_user.weight
        items_embedding=self.embed_item.weight

        gcn1_users_embedding = torch.sparse.mm(self.user_item_matrix, items_embedding) #+ users_embedding.mul(self.d_i_train))#*2. #+ users_embedding
        gcn1_items_embedding = torch.sparse.mm(self.item_user_matrix, users_embedding) #+ items_embedding.mul(self.d_j_train))#*2. #+ items_embedding

        gcn2_users_embedding = torch.sparse.mm(self.user_item_matrix, gcn1_items_embedding) #+ gcn1_users_embedding.mul(self.d_i_train))#*2. + users_embedding
        gcn2_items_embedding = torch.sparse.mm(self.item_user_matrix, gcn1_users_embedding) #+ gcn1_items_embedding.mul(self.d_j_train))#*2. + items_embedding
          
        gcn3_users_embedding = torch.sparse.mm(self.user_item_matrix, gcn2_items_embedding) #+ gcn2_users_embedding.mul(self.d_i_train))#*2. + gcn1_users_embedding
        gcn3_items_embedding = torch.sparse.mm(self.item_user_matrix, gcn2_users_embedding) #+ gcn2_items_embedding.mul(self.d_j_train))#*2. + gcn1_items_embedding

        gcn_users_embedding = users_embedding + (1/2)*gcn1_users_embedding + (1/3)*gcn2_users_embedding + (1/4)*gcn3_users_embedding
        gcn_items_embedding = items_embedding + (1/2)*gcn1_items_embedding + (1/3)*gcn2_items_embedding + (1/4)*gcn3_items_embedding

        return gcn_users_embedding, gcn_items_embedding, [users_embedding, gcn1_users_embedding]

def readD(set_matrix,num_):
    user_d=[]
    for i in range(num_):
        len_set=1.0/(len(set_matrix[i])+1)  
        user_d.append(len_set)
    return user_d

def readTrainSparseMatrix(set_matrix,u_d,i_d,is_user,is_train=1):
    user_items_matrix_i=[]
    user_items_matrix_v=[]
    if is_user:
        d_i=u_d
        d_j=i_d
    else:
        d_i=i_d
        d_j=u_d
    for i in set_matrix:
        # len_set=len(set_matrix[i])
        for j in set_matrix[i]:
            user_items_matrix_i.append([i,j])
            if is_train:
                d_i_j=np.sqrt(d_i[i]*d_j[j])
            else:
                d_i_j=d_i[i]
            #1/sqrt((d_i+1)(d_j+1))
            user_items_matrix_v.append(d_i_j)#(1./len_set) 

    # user_items_matrix_i=torch.cuda.LongTensor(user_items_matrix_i)
    # user_items_matrix_v=torch.cuda.FloatTensor(user_items_matrix_v)
    user_items_matrix_i=torch.tensor(user_items_matrix_i, dtype=torch.long, device='cuda')
    user_items_matrix_v=torch.tensor(user_items_matrix_v, dtype=torch.float, device='cuda')
    # return torch.sparse.FloatTensor(user_items_matrix_i.t(), user_items_matrix_v)
    return torch.sparse_coo_tensor(user_items_matrix_i.t(), user_items_matrix_v, dtype=torch.float, device='cuda')

class BPRData(data.Dataset):
    def __init__(self, train_dict=None, num_item=0, num_ng=1, is_training=None, data_set_count=0, all_rating=None):
        super(BPRData, self).__init__()
        
        self.num_item = num_item
        self.train_dict = train_dict
        self.num_ng = num_ng
        self.is_training = is_training
        self.data_set_count = data_set_count
        self.all_rating=all_rating
        self.set_all_item=set(range(num_item))  

    def ng_sample(self):
        # assert self.is_training, 'no need to sampling when testing'
        # print('ng_sample----is----call-----') 
        self.features_fill = []
        for user_id in self.train_dict:
            positive_list=self.train_dict[user_id]#self.train_dict[user_id]
            all_positive_list=self.all_rating[user_id]
            #item_i: positive item ,,item_j:negative item   
            # temp_neg=list(self.set_all_item-all_positive_list)
            # random.shuffle(temp_neg)
            # count=0
            # for item_i in positive_list:
            #     for t in range(self.num_ng):   
            #         self.features_fill.append([user_id,item_i,temp_neg[count]])
            #         count+=1
            for item_i in positive_list:
                for t in range(self.num_ng):
                    item_j=np.random.randint(self.num_item)
                    while item_j in all_positive_list:
                        item_j=np.random.randint(self.num_item)
                    self.features_fill.append([user_id,item_i,item_j])

    def __len__(self):  
        return self.num_ng*self.data_set_count#return self.num_ng*len(self.train_dict)


    def __getitem__(self, idx):
        features = self.features_fill
        
        user = features[idx][0]
        item_i = features[idx][1]
        item_j = features[idx][2]
        return user, item_i, item_j