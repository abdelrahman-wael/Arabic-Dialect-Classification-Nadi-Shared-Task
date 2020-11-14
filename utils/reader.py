import re
import pyarabic.araby as arb
import pandas as pd
import numpy as np

from utils.helper_func import *

# from preprocess_arabert import  preprocess
from transformers import AutoTokenizer
# from py4j.java_gateway import JavaGateway




labels_dictionary = {'Iraq':0,
'Egypt': 1,
'Algeria':2,
'Yemen':3,
'Saudcudai_Arabia':4,
'Syria':5,
'United_Arab_Emirates':6,
'Oman':7,
'Jordan':8,
'Tunisia':9,
'Kuwait':10,
'Morocco':11,
'Libya':12,
'Qatar':13,
'Lebanon':14,
'Sudan':15,
'Mauritania':16,
'Palestine':17,
'Somalia':18,
'Bahrain':19,
'Djibouti':20}



def bert_pre_processing(tweet,farasa):
  text_preprocessed = preprocess(tweet, do_farasa_tokenization=True , farasa=farasa)
  return text_preprocessed


def clean_tweet(tweet):
    result = re.sub(r"http\S+", "", tweet)
    result=re.sub(r"pic\S+", "", result)
    result = re.sub(r"@\S+","",result)
    result = arb.strip_tashkeel(result)
    return result


def read_tweets(file_path,file_name,farasa):
    """
    function to read labeled training set or dev set and output dataframe
    """
    
    df = pd.read_csv( file_path,dtype="string")
    df["tweet"]=df['tweets']
    df['preprocessed tweet'] = df['tweets'].apply(lambda x : bert_pre_processing(x,farasa) )
    df['label'] = df['labels'].apply(lambda x : labels_dictionary.get(x))

    df.to_csv("valid.csv")
    return df

def read_csv(file_path,upsampled=False):
  
  '''
  read pre-porcessed file
  '''
  csv=pd.read_csv(file_path)
  csv.dropna(inplace=True)
  if upsampled:
    csv=upsampling(csv)
  tweets = list(csv["preprocessed tweet"])
  labels = list(csv["label"])
  
  return (tweets,labels)


def read_tweets_unlabeled(path):
    """
    function to read unlabeled data and output a dataframe
    """
    filenames = glob.glob(os.path.join(path, "*.csv"))
    li = []

    for file in filenames:
        df = pd.read_csv(file)
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)
    del li
    frame = frame.dropna()
    return frame

