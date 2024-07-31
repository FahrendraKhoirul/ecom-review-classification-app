import torch.nn as nn
import torch
from huggingface_hub import PyTorchModelHubMixin

class IndoBertEcommerceReview(nn.Module, PyTorchModelHubMixin):
    def __init__(self, bert):
        super().__init__()
        self.bert  = bert
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = self.sigmoid(logits)
        return probabilities
        
class IndoBertCNNEcommerceReview(nn.Module, PyTorchModelHubMixin):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.conv1 = nn.Conv1d(in_channels=bert.config.hidden_size, out_channels=512, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(512, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # Permute to [batch_size, hidden_size, seq_len]
        last_hidden_state = last_hidden_state.permute(0, 2, 1)

        conv1_output = self.conv1(last_hidden_state)
        pooled_output = self.pool(conv1_output).squeeze(-1)
        logits = self.linear(pooled_output)
        probabilities = self.sigmoid(logits)
        return probabilities

class IndoBertLSTMEcommerceReview(nn.Module, PyTorchModelHubMixin):
    def __init__(self, bert):
      super().__init__()
      self.bert = bert
      self.lstm = nn.LSTM(bert.config.hidden_size, 128)
      self.linear = nn.Linear(128, 3)
      self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
      outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
      last_hidden_state = outputs.last_hidden_state
      lstm_out, _ = self.lstm(last_hidden_state)
      pooled = lstm_out[:, -1, :]
      logits = self.linear(pooled)
      probabilities = self.sigmoid(logits)
      return probabilities
    
