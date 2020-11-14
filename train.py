from transformers import get_linear_schedule_with_warmup,AdamW
from transformers import BertForSequenceClassification
from utils.helper_func import format_time

from utils.tokenizer import Tokenizer
from data_loader import create_bert_dataloader
from utils.reader import read_csv

import torch.nn as nn
from models import get_model
import json
import pandas as pd
import time
import datetime
import random
import numpy as np
import torch
from run_model import run_model
import os
from utils.helper_func import *

def get_loss_weights(train_labels):

  unique,count=np.unique(train_labels,return_counts=True)
  
  weights=[1-freq/len(train_labels) for freq in count]
          
  return weights


def save_model(model,model_name,model_path,f1_score,accuracy):
  parent_dir=os.getcwd()
  os.chdir(model_path)
  folder_name = model_name
  all_files = os.listdir()
  if folder_name not in all_files:
    os.mkdir(str(folder_name))
  os.chdir(folder_name)
  torch.save(model,"best_validation "+str(f1_score))
  os.chdir(parent_dir)
  



def train(train_loader,valid_loader, epochs=20
          ,learning_rate=2e-5,regularization = 0.01
          ,eps=1e-8,model=None,device="cuda"
          , loss_weights =None,loss_func=None 
          ,save_path="save models"
          ,model_name=None):
  
  format_time(time.time()-time.time())
  
  
  model.to(device)
  
  optimizer = AdamW(model.parameters(),
                    lr = learning_rate, 
                    eps = eps)
  

  total_steps = len(train_loader) * epochs

  scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)



  seed_val = 42

  random.seed(seed_val)
  np.random.seed(seed_val)
  torch.manual_seed(seed_val)
  torch.cuda.manual_seed_all(seed_val)

  # We'll store a number of quantities such as training and validation loss, 
  # validation accuracy, and timings.
  training_stats = []

  # Measure the total training time for the whole run.
  total_t0 = time.time()

  # if loss func not specified use model 's own loss
  if loss_func == "weighted_CrossEntropy":
    print("using weighted_CrossEntropy loss")
    # loss_func=nn.CrossEntropyLoss(weight=loss_weights,size_average=False)
    loss_func=nn.CrossEntropyLoss()
  
  best_valid_f1 = 0
  best_valid_preds =[]
  # For each epoch...
  for epoch_i in range(epochs):
      
      # ========================================
      #               Training
      # ========================================
      
      # Perform one full pass over the training set.

      print("")
      print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
      # print('Training...')
      model.to(device)

      # Measure how long the training epoch takes.
      t0 = time.time()

      training_loss,training_acc,training_f1,training_recall,training_preds,training_labels=run_model(model,train_loader,True,optimizer,scheduler,device=device,loss_func=loss_func)
      print("  Average training loss: {:.6f}".format(training_loss))
      print("  Average training accuracy: {0:.4f}".format(training_acc))
      print("  Average training f1: {0:.4f}".format(training_f1))
      print("  Average training recall: {0:.4f}".format(training_recall))
      print("-"*50)

      valid_loss,valid_acc,valid_f1,valid_recall,valid_preds,valid_labels = run_model(model,valid_loader,device=device,loss_func=loss_func)

      print("  Average validation loss: {0:.4f}".format(valid_loss))
      print("  Average validation accuracy: {0:.4f}".format(valid_acc))
      print("  Average validation f1: {0:.4f}".format(valid_f1))
      print("  Average validation recall: {0:.4f}".format(valid_recall))
      print("-"*50)
      
      # Measure how long this epoch took.
      training_time = format_time(time.time() - t0)

      # if valid_f1 > best_valid_f1 :
      #   save_model(model,model_name,save_path,valid_f1,valid_acc)
      #   best_valid_preds =valid_preds 
      #   save_pred_txt(best_valid_preds)
        


def main():  

  with open("train_config.txt", "r") as read_file:
    config_dic = json.load(read_file) 
  
  train_tweets , train_labels = read_csv("data/preprocessed data/labeled training.csv",True)
  valid_tweets , valid_labels = read_csv("data/preprocessed data/labeled valid.csv")
  
  
 
  loss_weights=get_loss_weights(train_labels)

  tokenizer = Tokenizer()
  train_data = tokenizer.bert_tokenize_data(train_tweets,train_labels)
  valid_data = tokenizer.bert_tokenize_data(valid_tweets , valid_labels)


  train_loader,valid_loader=create_bert_dataloader(train_data,valid=valid_data,
                                                  batch_size=config_dic["batch_size"],
                                                  split_train= config_dic["split_train"],
                                                  test_size=config_dic["split_size"])

  model=get_model(name=config_dic["model"],path=config_dic["pretrained_path"])
  
  train(train_loader,
        valid_loader,
        model=model,
        epochs=config_dic["epochs"],
        learning_rate=config_dic["learning_rate"],
        eps=config_dic["eps"],
        device = config_dic["device"],
        loss_func=config_dic["loss_func"],
        save_path = config_dic["save_model_path"],
        model_name=config_dic["model"]
        )

if __name__=="__main__":
  main()
