import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F


class BertForSentiment(nn.Module):
  def __init__(self, bert_model_name='bert-base-uncased'):
    super(BertForSentiment, self).__init__()
    self.bert = BertModel.from_pretrained(bert_model_name)
    self.out = nn.Sequential(
      nn.Linear(768, 256),
      nn.ReLU(),
      nn.Linear(256, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, 1),
      nn.ReLU()
    )

    for param in self.bert.parameters():
      param.requires_grad = False

  def forward(self, input_ids, attention_mask=None):
    outputs = self.bert(input_ids, attention_mask=attention_mask)
    last_hidden_state_cls = outputs[0][:, 0, :]
    out = self.out(last_hidden_state_cls)
    return out