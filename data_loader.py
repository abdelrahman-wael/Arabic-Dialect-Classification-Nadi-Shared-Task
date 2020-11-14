from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch

def create_bert_dataloader(train,valid=None,test_size=0.05,batch_size = 32,split_train=True):
  
  
  if split_train:
    tweets , labels = train
    train_tweets, valid_tweets,train_labels,  valid_labels = train_test_split(tweets, labels, test_size=test_size, random_state=42,stratify=labels)
  else:
    train_tweets, train_labels = train
    valid_tweets , valid_labels =valid
  

  train_input,train_mask = ( [ input_ for input_,mask in train_tweets],
                            [mask for input_,mask in train_tweets]  )
  valid_input,valid_mask = ( [ input_ for input_,mask in valid_tweets],
                            [mask for input_,mask in valid_tweets]  )
  # return train_input,train_mask

  # transfrom lists to tensors
  # return train_input,train_mask,train_labels

  train_input,train_mask = [torch.cat(train_input, dim=0)
                            ,torch.cat(train_mask, dim=0) ]
  train_labels = torch.tensor(train_labels)
  
  valid_input,valid_mask = [torch.cat(valid_input, dim=0)
                            ,torch.cat(valid_mask, dim=0) ]
  valid_labels = torch.tensor(valid_labels)
  
  
  

  train_dataset = TensorDataset(train_input,train_mask,train_labels)
  val_dataset = TensorDataset(valid_input,valid_mask,valid_labels)
  
  
  train_dataloader = DataLoader(
              train_dataset,  # The training samples.
              sampler = RandomSampler(train_dataset), # Select batches randomly
              batch_size = batch_size # Trains with this batch size.
          )

  validation_dataloader = DataLoader(
              val_dataset, # The validation samples.
              sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
              batch_size = batch_size # Evaluate with this batch size.
          )
  
  return train_dataloader,validation_dataloader
