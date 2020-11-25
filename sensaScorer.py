#Imports

import torch
from tqdm.notebook import tqdm

from transformers import BertTokenizer
from torch.utils.data import TensorDataset

from transformers import BertForSequenceClassification

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from google_drive_downloader import GoogleDriveDownloader as gdd

import seaborn as sns
import torch
import warnings

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

class SensaScorer():
  def __init__(self):
    # self.sensa_dict = {1:'Barely sensationalist',0: 'Not sensationalist',2:'Sensationalist'}
    self.sensa_dict = {1:0.55,0:0.25,2:0.95}
    self.label_dict = {'Barely sensationalist': 1, 'Not sensationalist': 0, 'Sensationalist': 2}
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                              num_labels=len(self.label_dict),
                                                              output_attentions=False,
                                                              output_hidden_states=False)
    self.model.to(self.device)
    gdd.download_file_from_google_drive(file_id='1SpfmiCq2a2aXTXvFW6cHnm-0eBCpcyxY',
                                  dest_path='./sensationalism_BERT_best.model',
                                  unzip=False)
    
  def getScore(self,title):
    self.model.load_state_dict(torch.load('sensationalism_BERT_best.model', 
                                      map_location=torch.device('cpu')))
    prediction = self.evaluateSentimentScore(title)
    
    return prediction

  #1 Function that receives a String with max 250 characters
  def evaluateSentimentScore(self,news_title):
    #Add title string to dataframe
    df = pd.DataFrame(columns=['English','label'])
    df.loc[0]=[news_title,0]
    #Instantiate BERT Tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', 
                                              do_lower_case=True)
    #Econded record(s)
    encoded_data = tokenizer.batch_encode_plus(
      df.English.values, 
      add_special_tokens=True, 
      return_attention_mask=True, 
      pad_to_max_length=True, 
      max_length=256, 
      return_tensors='pt'
      )
    #Get inputs_ids_val and attention_masks_val
    input_ids = encoded_data['input_ids']
    attention_masks = encoded_data['attention_mask']
    labels = torch.tensor([0])
    #Tensor Dataset
    dataset = TensorDataset(input_ids, attention_masks, labels)
    #Create dataloader
    dataloader = DataLoader(dataset,
                            sampler=SequentialSampler(dataset),
                            batch_size=1)
    #Call evalation method and return values
    return self.evaluate(dataloader)

  # 2 Define Evaluation method Similar to evaluate but 
  def evaluate(self,dataloader):
    self.model.eval()

    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in dataloader:
        
        batch = tuple(b.to(self.device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                  }

        with torch.no_grad():        
            outputs = self.model(**inputs)
            
        logits = outputs[1]
        logits = logits.detach().cpu().numpy()
        predictions.append(logits)
    
    preds = np.concatenate(predictions, axis=0)
    preds_flat = np.argmax(preds, axis=1).flatten()
    return self.sensa_dict[preds_flat[0]]
