import numpy as np
import time
import datetime
from sklearn.utils import resample
import pandas as pd 

keys_dictionary = {0:'Iraq',
1: 'Egypt',
2: 'Algeria',
3: 'Yemen',
4: 'Saudi_Arabia',
5: 'Syria',
6: 'United_Arab_Emirates',
7: 'Oman',
8: 'Jordan',
9: 'Tunisia',
10: 'Kuwait',
11: 'Morocco',
12: 'Libya',
13: 'Qatar',
14: 'Lebanon',
15: 'Sudan',
16: 'Mauritania',
17: 'Palestine',
18: 'Somalia',
19: 'Bahrain',
20: 'Djibouti'}


def flat_accuracy(logits, labels,device):
  pred_flat = np.argmax(logits.cpu().detach().numpy(), axis=1).flatten()
  labels_flat = labels.cpu().detach().numpy().flatten()
  return np.sum(pred_flat == labels_flat) / len(labels_flat)

def upsampling(data,target_size= 750):
  data_resampled = data
  for i in range(21):
    class_upsampled = data[data['label'] == i ]
    if len(class_upsampled) < target_size:
      class_upsampled = resample(class_upsampled, replace=True, n_samples=750 - len(class_upsampled))
      data_resampled = pd.concat([data_resampled,class_upsampled])
  return data_resampled

from sklearn.metrics import classification_report

def get_report(predictions,true_labels):
  pred = [item for sublist in predictions for item in sublist]

  true_label = [item for sublist in true_labels for item in sublist]

  prediction = []
  for i in range(len(pred)):
    prediction.append(np.argmax(pred[i], axis=0).flatten()[0])

  print(classification_report(true_label, prediction,target_names= list(labels_dictionary.keys())))
  return prediction

def save_pred_txt(predictions):
  pred_labels = []
  for item in predictions:
    pred_labels.append(keys_dictionary.get(item))

  with open('dev.txt', 'w') as f:
    for item in pred_labels:
        f.write("%s\n" % item)

def get_preds(logits,device):
  preds = np.argmax(logits.cpu().detach().numpy(), axis=1).flatten()
  return preds.tolist()

def get_labels(labels,device):
  labels = labels.cpu().detach().numpy().flatten()
  return labels.tolist()

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))