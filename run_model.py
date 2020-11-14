import time
from utils.helper_func import *
import torch
from sklearn.metrics import f1_score,accuracy_score,recall_score
import torch.nn as nn



def run_model(model,data_loader,train=False,optimizer=None,
              scheduler=None,device="cuda", loss_func=None):

  if train:
    model.train()
  else :
    model.eval()

  # Reset the total loss for this epoch.

  total_loss = 0
  total_accuracy=0
  total_f1 =0
  t0 = time.time()
  all_preds =[]
  all_labels =[]
  for step, batch in enumerate(data_loader):

          # Progress update every 40 batches.
          if step % 40 == 0 and not step == 0:
              # Calculate elapsed time in minutes.
              elapsed = format_time(time.time() - t0)
              
              # Report progress.
              print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(data_loader), elapsed))

          # Unpack this training batch from our dataloader. 

          # `batch` contains three pytorch tensors:
          #   [0]: input ids 
          #   [1]: attention masks
          #   [2]: labels 
          input_ids = batch[0].to(device)
          input_mask = batch[1].to(device)
          labels = batch[2].to(device)

          model.zero_grad()        

          
          model_loss, logits = model(input_ids, 
                              token_type_ids=None, 
                              attention_mask=input_mask, 
                              labels=labels)

 
          

          total_accuracy += flat_accuracy(logits, labels,device)
          
          
          all_preds += get_preds(logits,device)
          all_labels += get_labels(labels,device)
          if loss_func:
            loss=loss_func(logits.view(-1,21) 
                              ,labels.view(-1))
            total_loss +=loss 
            loss.backward()
            
          else:
            total_loss += model_loss  
            model_loss.backward()      
          
          


          # Perform a backward pass to calculate the gradients.
          

          # Clip the norm of the gradients to 1.0.
          # This is to help prevent the "exploding gradients" problem.
          torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

          if train:
            optimizer.step()
            # Update the learning rate.
            scheduler.step()

  avg_loss = total_loss / len(data_loader)            
  avg_acc = total_accuracy / len(data_loader)
  avg_f1 = f1_score(all_labels,all_preds,average='macro')
  recall = recall_score(all_labels,all_preds,average='macro')

  return avg_loss,avg_acc,avg_f1,recall,all_preds,all_labels
          
