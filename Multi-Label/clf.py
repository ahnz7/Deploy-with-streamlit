#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   clf.py
@Time    :   2021/03/18 21:49:48
@Author  :   Hanlin Li 
@Version :   1.0
@Contact :   ahnz830@gmail.com
'''

# here put the import lib

import torch
from torch import nn 
import numpy as np 
import pandas as pd 
import os 
import json
from transformers import WEIGHTS_NAME, CONFIG_NAME
from transformers import AlbertModel,AlbertTokenizer,get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import openpyxl
from argparse import ArgumentParser
import argparse
import logging
from apex import amp
import random
from datetime import datetime

def parse_arguments():          #input
    parser = ArgumentParser()
    parser.add_argument('--seed',type = int,help="seed value")
    parser.add_argument('--num_epochs',type = int,help="Training epochs")
    parser.add_argument('--warmup_steps',type = int,help="Warm up steps",default=10 ** 3)
    parser.add_argument('--lr',type = float,help="learning rate",default=2e-5)
    parser.add_argument('--eps',type = float,help="epsilon",default=1e-8)
    parser.add_argument('--input_file',type=dir_file, required=True,help = 'The input data file. Should contain the  .json files, which are needed to clean')
    parser.add_argument('--output_dir',type=dir_file, required=True,help = 'The output data dir. Should contain the  .json files, which are needed to clean')
    return parser.parse_args()

def dir_file(path):
    if os.path.isfile(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid file")

def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")


def set_seed(seed):                         #reproducible
    """ Set all seeds to make results reproducible """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def data_works(path):                           #read the excel 
    dataset = pd.read_excel(path,engine = 'openpyxl')
    dataset.drop(columns=['LABELS'],inplace=True)
    dataset.rename(columns={'Unnamed: 0':'TEXT_IDS'},inplace=True)
    dataset.set_index(['TEXT_IDS'],inplace=True)
    return dataset

class CustomDataset(Dataset):               #set the Dataset
    def __init__(self,tokenizer,data):
        self.tokenizer = tokenizer
        self.labels = data[[0,1,2,3,4,5,6]].values  #0~6 are symbol for "aim of study","Patients","device" ...This can be set to any value
        self.data = data                
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        encoded_pair = self.tokenizer(self.data['CONTEXTS'].tolist(),truncation=True,padding = True,return_tensors='pt')
        token_ids = encoded_pair['input_ids'].squeeze(0)
        attn_masks = encoded_pair['attention_mask'].squeeze(0)
        token_type_ids = encoded_pair['token_type_ids'].squeeze(0)
        label = torch.Tensor(self.data[[0,1,2,3,4,5,6]].values)
        return (token_ids[idx],attn_masks[idx],token_type_ids[idx],label[idx])

class MyBert(nn.Module):
    def __init__(self, freeze_bert=False, model = AlbertModel.from_pretrained('albert-base-v2'), num_classes = 7):
        super(MyBert,self).__init__()
        self.bert = model
        if freeze_bert:                         # our dataset is small, so we just need to train the Parameters in top layer
            for p in self.bert.parameters():
                p.requires_grad = False
        self.num_classes = num_classes
        self.dropout = nn.Dropout(p = 0.5)
        self.classifier = nn.Linear(768,self.num_classes)
    
    def forward(self,input_ids = None,attention_mask = None,token_type_ids = None):
        outputs = self.bert(input_ids,attention_mask,token_type_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
#         loss_fct = nn.BCEWithLogitsLoss()
#         loss = loss_fct(logits.view(-1, 7), labels.view(-1,7))
        return logits

def train(model,dl,loss_fct,scheduler,optimizer,device,epoch):
    model.train()                           
    total_loss = 0
    
    print("\n Start Training...")
    nowtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("=========="*8 + "%s"%nowtime)
    for step, batch in enumerate(dl):
        torch.cuda.empty_cache()
        optimizer.zero_grad()

        batch = tuple(t.to(device,non_blocking = True) for t in batch)
        logits = model(batch[0],batch[1],batch[2])                      #inputs_ids,attention,token_type
        loss = loss_fn(logits.view(-1,7),batch[3].view(-1,7))           #size:[batch_size,labels->7]
        total_loss += loss.item()

        
        with amp.scale_loss(loss, optimizer) as scaled_loss:        #Here fp16 is used
            scaled_loss.backward()
        # loss.backward()
        optimizer.step()
        scheduler.step()
        if step%100 == 0:
            print(("[step = %d loss_aver: %.3f,")%(step,total_loss/(step+1)))
    print('EPOCH {:0d}: Training loss_aver: {:.3f}'.format(epoch+1,total_loss/step))

def evaluate(model, dl,loss_fct):
    model.eval()
    pred = []
    true = []
    logger.info("\n Start evaluating...")
    logger.info("=========="*8)
    with torch.no_grad():
        total_loss = 0
        for i, batch in enumerate(dl):
            batch = tuple(arr.to(device) for arr in batch)
            logits = model(batch[0],batch[1],batch[2])
            loss = loss_fct(logits.view(-1,7),batch[3].view(-1,7))
            total_loss += loss
            true += batch[3].cpu().numpy().tolist()
            pred += logits.cpu().numpy().tolist()
    true = np.array(true)
    pred = np.array(pred)
    # for i, name in enumerate([0,1,2,3,4,5]):              #not decide which metric to be used
    #     logger.info(f"{name} roc_auc {roc_auc_score(true[:, i], pred[:, i])}")
    logger.info(f"Evaluate loss {total_loss / len(dl)}")
    logger.info('Evalution finish ...')


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    args = parse_arguments()
    
    set_seed(args.seed) # Set all seeds to make results reproducible

    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info('Start...')
    
    dataset = data_works(args.input_file)  
    train_df,val_df = train_test_split(dataset,test_size = 0.2,shuffle = True)
    logger.info('Reading training data...')
    
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    
    train_ds,val_ds = CustomDataset(tokenizer,train_df),CustomDataset(tokenizer,val_df)
    train_dl,val_dl = DataLoader(train_ds,batch_size= 8,shuffle=True,num_workers = 8,pin_memory = True),DataLoader(val_ds,batch_size= 8,shuffle=True,num_workers = 8,pin_memory = True)
    
    model = MyBert(AlbertModel.from_pretrained('albert-base-v2')).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    EPOCH_NUM = args.num_epochs
    
    warmup_steps = args.warmup_steps            # triangular learning rate, linearly grows untill half of first epoch, then linearly decays 
    total_steps = len(train_dl) * EPOCH_NUM - warmup_steps
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.eps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    
    logger.info("***** Running training *****")
    for i in range(EPOCH_NUM):
        print('=' * 50, f"EPOCH {i}", '=' * 50)
        train(model, train_dl,loss_fn, scheduler,optimizer,device,i)
        evaluate(model, val_dl,loss_fn)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, args.output_dir)
    print('****** Model is saved ******')
