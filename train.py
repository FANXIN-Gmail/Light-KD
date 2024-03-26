# -- coding:UTF-8
import torch
# print(torch.__version__)
import torch.nn as nn 

import wandb

import argparse
import os
import numpy as np
import math
import sys
import random

# print(torch.cuda.is_available())
# print(torch.cuda.device_count())
# print(torch.cuda.current_device())

CUDA_VISIBLE_DEVICES = 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print(torch.cuda.get_device_name(CUDA_VISIBLE_DEVICES))

os.environ["WANDB_API_KEY"] = "15b5e8572b0516899bae70d5bbb5c9091d1667a7"

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn.functional as F
import torch.autograd as autograd
from sklearn.cluster import KMeans

import pdb
from collections import defaultdict
import time
import collections
from shutil import copyfile

from evaluate import *
from data_utils import *
import C

wandb.login()

dataset_base_path='/data/fan_xin/Gowalla'

epoch_num=C.EPOCH

user_num=C.user_num
item_num=C.item_num

factor_num=256
batch_size=1024*4
learning_rate=C.LR

num_negative_test_val=-1##all

run_id=C.RUN_ID
print(run_id)
dataset='Gowalla'

path_save_model_base='/data/fan_xin/newlossModel_PIW/'+dataset+'/'+run_id
if (os.path.exists(path_save_model_base)):
    print('has model save path')
else:
    os.makedirs(path_save_model_base)

base = read(dataset_base_path + "/check_in.json", [0, 0.6])
block = read(dataset_base_path + "/check_in.json", [0.6, 0.7])
training_user_set, training_item_set = list_to_set(block)
training_user_set_, training_item_set_ = list_to_set(base)
training_set_count = count_interaction(training_user_set)
user_rating_set_all = json_to_set(dataset_base_path + "/check_in.json", single=1)

# Positive sampling in old graph
u_sample = list()
i_sample = list()

for u in training_user_set_:
    i = random.choice(list(training_user_set_[u]))
    u_sample.append(u)
    i_sample.append(i)

for i in training_item_set_:
    u = random.choice(list(training_item_set_[i]))
    u_sample.append(u)
    i_sample.append(i)

u_sample = torch.tensor(u_sample).cuda()
i_sample = torch.tensor(i_sample).cuda()

print(training_set_count)

training_user_set[user_num-1].add(item_num-1)
training_item_set[item_num-1].add(user_num-1)
training_user_set_[user_num-1].add(item_num-1)
training_item_set_[item_num-1].add(user_num-1)

u_d=readD(training_user_set,user_num)
i_d=readD(training_item_set,item_num)
u_d_=readD(training_user_set_,user_num)
i_d_=readD(training_item_set_,item_num)

sparse_u_i=readTrainSparseMatrix(training_user_set,u_d,i_d,True)
sparse_i_u=readTrainSparseMatrix(training_item_set,u_d,i_d,False)
sparse_u_i_=readTrainSparseMatrix(training_user_set_,u_d_,i_d_,True)
sparse_i_u_=readTrainSparseMatrix(training_item_set_,u_d_,i_d_,False)

local_u_i = readTrainSparseMatrix(training_user_set_,u_d_,i_d_,True,True)
local_i_u = readTrainSparseMatrix(training_item_set_,u_d_,i_d_,False,True)

train_dataset = BPRData(
        train_dict=training_user_set, num_item=item_num, num_ng=5, is_training=True,\
        data_set_count=training_set_count, all_rating=user_rating_set_all)
train_loader = DataLoader(train_dataset,
        batch_size=batch_size, shuffle=True, num_workers=0)

PATH_model='/data/fan_xin/newlossModel_PIW/'+dataset+'/'+C.BASE+'/epoch'+str(C.BASE_EPOCH)+'.pt'

model_ = BPR(user_num, item_num, factor_num, sparse_u_i_, sparse_i_u_).to('cuda')
model_.load_state_dict(torch.load(PATH_model))
model_.eval()
with torch.no_grad():
    old_U_emb, old_I_emb, old_User = model_.inference() 

model = BPR(user_num,item_num,factor_num,sparse_u_i,sparse_i_u,
    local_u_i=local_u_i,
    local_i_u=local_i_u,
    old_U_emb=old_U_emb,
    old_I_emb=old_I_emb,
    old_User=old_User,
    u_sample=u_sample,
    i_sample=i_sample
    ).to('cuda')
model.load_state_dict(torch.load(PATH_model))
model.extra_define(factor_num)
model.to('cuda')

optimizer_bpr = torch.optim.Adam(model.parameters(), lr=learning_rate)#, betas=(0.5, 0.99))

run = wandb.init(
    # Set the project where this run will be logged
    project="KD-PIW-Gowalla",
    # notes="random_without_remap",
    # tags=["ramdom", "10%"],
    name=run_id,
    mode="offline",
)

########################### TRAINING #####################################

# testing_loader_loss.dataset.ng_sample()

print('--------training processing-------')
count, best_hr = 0, 0
for epoch in range(epoch_num):

    model.train() 
    start_time = time.time()

    train_loader.dataset.ng_sample()

    # pdb.set_trace()
    print('train data of ng_sample is  end')
    # elapsed_time = time.time() - start_time
    # print(' time:'+str(round(elapsed_time,1)))
    # start_time = time.time()

    train_loss_sum=[]

    for user, item_i, item_j in train_loader:

        user = user.cuda()
        item_i = item_i.cuda()
        item_j = item_j.cuda()

        model.zero_grad()
        loss = model(user, item_i, item_j)
        loss.backward()
        optimizer_bpr.step()
        count += 1
        train_loss_sum.append(loss.item())

    elapsed_time = time.time() - start_time
    train_loss=round(np.mean(train_loss_sum[:-1]),4)

    str_print_train="epoch:"+str(epoch)+' time:'+str(round(elapsed_time,1))+'\t train loss:'+str(train_loss)
    # print('--train--',elapsed_time)

    wandb.log({"train_loss": train_loss})

    print(str_print_train)

    PATH_model=path_save_model_base+'/epoch'+str(epoch)+'.pt'
    torch.save(model.state_dict(), PATH_model)
