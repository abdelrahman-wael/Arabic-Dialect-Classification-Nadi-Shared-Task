from transformers import AutoTokenizer
import torch

# creating tokenizer class uses bert by default


class Tokenizer():
  def __init__(self,tokenizer_type="bert",upsample=False):
    self.tokenizer_type=tokenizer_type
    if tokenizer_type =="bert":
      self.tokenizer = AutoTokenizer.from_pretrained("/content/drive/My Drive/our repo NADI shared task/NADI_Shared_Task/pretrained bert") 

  def bert_tokenize_tweet(self,sent):
    encoded_dict = self.tokenizer.encode_plus(
                          sent,                      # Sentence to encode.
                          add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                          max_length = 64,           # Pad & truncate all sentences.
                          pad_to_max_length = True,
                          return_attention_mask = True,   # Construct attn. masks.
                          return_tensors = 'pt',     # Return pytorch tensors.
                    )
    return encoded_dict['input_ids'],encoded_dict['attention_mask']
    

  

  def bert_tokenize_data(self,text,labels):

    input_ids = []
    attention_masks = []

    for tweet in text:
      input_id,atten_mask=self.bert_tokenize_tweet(tweet)
      input_ids.append(input_id)
      attention_masks.append(atten_mask)

    tweets = list(zip(input_ids,attention_masks))

    data = (tweets,labels)
    

    return data
    