import torch.nn as nn
import torch
import numpy as np
from huggingface_hub import PyTorchModelHubMixin
from transformers import BertModel, AutoTokenizer

class IndoBertLSTMEcommerceReview(nn.Module, PyTorchModelHubMixin):
    def __init__(self, bert):
      super().__init__()
      self.bert = bert
      self.lstm = nn.LSTM(bert.config.hidden_size, 128)
      self.linear = nn.Linear(128, 3)
      self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
      outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
      # print(outputs.keys())
      last_hidden_state = outputs.last_hidden_state
      lstm_out, _ = self.lstm(last_hidden_state)
      pooled = lstm_out[:, -1, :]
      logits = self.linear(pooled)
      probabilities = self.sigmoid(logits)
      return probabilities

bert = BertModel.from_pretrained("indobenchmark/indobert-base-p1")
tokenizer = AutoTokenizer.from_pretrained("fahrendrakhoirul/indobert-finetuned-ecommerce-reviews")
        
indobertlstm_model = IndoBertLSTMEcommerceReview.from_pretrained("fahrendrakhoirul/indobert-lstm-finetuned-ecommerce-reviews", bert=bert).to('cpu')

# run modell
res_token = tokenizer("hahahah",  return_tensors="pt").to('cpu')
input_ids = res_token['input_ids']  # Unpack dictionary
attention_mask = res_token['attention_mask']# Unpack dictionary

print(res_token)
with torch.no_grad():
    logits = indobertlstm_model(input_ids=input_ids, attention_mask=attention_mask)
    preds = torch.sigmoid(logits).detach().cpu().numpy()[0]

print(preds)