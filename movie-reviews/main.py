import torch
from transformers import BertTokenizer
import re
from models import BertForSentiment

def text_preprocessing(text):
  """
  - Remove entity mentions (eg. "@united")
  - Correct errors (eg. "&amp;" to "&")
  @param    text (str): a string to be processed.
  @return   text (Str): the processed string.
  """
  # Remove "@name"
  text = re.sub(r"(@.*?)[\s]", " ", text)

  # Replace "&amp;" with "&"
  text = re.sub(r"&amp;", "&", text)

  # Remove trailing whitespace
  text = re.sub(r"\s+", " ", text).strip()

  return text

MAX_LEN = 64
def preprocessing_for_bert(data):
  """Perform required preprocessing steps for pretrained BERT.
  @param    data (np.array): Array of texts to be processed.
  @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
  @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                tokens should be attended to by the model.
  """
  # Create empty lists to store outputs
  input_ids = []
  attention_masks = []

  # For every sentence...
  for sent in data:
    # `encode_plus` will:
    #    (1) Tokenize the sentence
    #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
    #    (3) Truncate/Pad sentence to max length
    #    (4) Map tokens to their IDs
    #    (5) Create attention mask
    #    (6) Return a dictionary of outputs
    encoded_sent = tokenizer.encode_plus(
      text=text_preprocessing(sent),  # Preprocess sentence
      add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
      max_length=MAX_LEN,             # Max length to truncate/pad
      padding="max_length",           # Pad sentence to max length
      return_attention_mask=True,     # Return attention mask,
      return_tensors="pt",            # Return pytorch tensors,    
      truncation=True
    )
    
    # Add the outputs to the lists
    input_ids.append(encoded_sent.get("input_ids"))
    attention_masks.append(encoded_sent.get("attention_mask"))


  return input_ids, attention_masks


if __name__ == "__main__":

  tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

  input_ids, attention_masks = preprocessing_for_bert(["Hello world", "This is super cool!"])

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  model = BertForSentiment().to(device)

  input_ids = torch.concat(input_ids, dim=0).to(device)
  attention_masks = torch.concat(attention_masks, dim=0).to(device)

  print(input_ids.shape)

  pred = model(input_ids, attention_masks)

  print(pred)