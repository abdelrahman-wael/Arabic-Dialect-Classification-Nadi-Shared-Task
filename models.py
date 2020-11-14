from transformers import BertForSequenceClassification
import torch
import torch.nn as nn
import torch
import numpy
import torch
import torch.nn as nn
from torch.autograd import Variable

from numpy import pi
import numpy as np


def get_model(name=None,path=None,freeze_bert=None,embedding=None):

  if name == "bert_classifier":
  
    model = get_bert_classifier(path)
    if freeze_bert:
      for param in model.bert.parameters():
        param.requires_grad = False
    return model

  if self.name == "bert_customized":
    print("using customized bert")
    model = bert_customized()
    return model


def get_bert_classifier(path=None):
  # if no saved model in path create a new one
  if path==None:
    return BertForSequenceClassification.from_pretrained(
        "aubmindlab/bert-base-arabertv01", # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 21, 
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
        )
  else:
    print("using fine tuned model ")
    return BertForSequenceClassification.from_pretrained(
        path, # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 21, 
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
        )


class bert_customized(torch.nn.Module):
  def __init__(self):
      
          super(bert_customized, self).__init__()

          bert = BertForSequenceClassification.from_pretrained(
          "/content/drive/My Drive/our repo NADI shared task/NADI_Shared_Task/pretrained bert", # Use the 12-layer BERT model, with an uncased vocab.
          num_labels = 21, 
          output_attentions = False, # Whether the model returns attentions weights.
          output_hidden_states = False, # Whether the model returns all hidden-states.
          ).bert

          self.bert=bert

          self.kernels = [3,5,7,9]
          device = "cuda"
          self.conv1 = [nn.Sequential(nn.Conv1d(1, 32,
                                                  kernel_size=k,
                                                  padding=0),
                                        nn.ReLU(),
                                        nn.MaxPool1d(3)).to(device) for k in self.kernels]                               
          self.conv2 = nn.Sequential(nn.Conv1d(32, 16, kernel_size=9, padding=0),
                                        nn.ReLU(),
                                        nn.MaxPool1d(3)
                                        )      
          self.fc1 = nn.Sequential(nn.Linear(5376,1024),
                                  nn.ReLU())
          self.fc2 = nn.Sequential(nn.Linear(1024,21),
                                  nn.ReLU())

  def forward(self, input_ids,token_type_ids=None, 
                              attention_mask=None, 
                              labels=None):
      
    hidden,pooled_output=self.bert(input_ids,attention_mask)

    x_out = []
    
    pooled_output = pooled_output.unsqueeze(1)
    # print(pooled_output.shape)
    for conv in self.conv1:
      
      x_out.append(conv(pooled_output))
      # print(x_out[-1].shape)
    x_out = torch.cat(x_out,dim=-1)
    # print(x_out.shape)

    x_out = self.conv2(x_out)
    # x_out = self.conv3(x_out)
    x_out = nn.Flatten()(x_out)
    # print(x_out.shape)
    x_out = self.fc1(x_out)
    logits = self.fc2(x_out)
    
    
  
    return 0 ,logits  
