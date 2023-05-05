import torch
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")
model.eval()

text = "sst2 sentence: it confirms fincher ’s status as a film maker who artfully bends technical know-how to the service of psychological insight"
with torch.no_grad():
  encoder_inputs = tokenizer(text, return_tensors="pt")
  decoder_input_ids = torch.tensor([tokenizer.pad_token_id]).unsqueeze(0) 
  print(tokenizer.pad_token_id)
  print(f"==>> decoder_input_ids: {decoder_input_ids}")
  outputs = model(**encoder_inputs, decoder_input_ids=decoder_input_ids)
  print(f"==>> outputs.: {outputs}")
  logits = outputs[0]
  print(f"==>> logits.shape: {logits.shape}")
  tokens = torch.argmax(logits, dim=2)
  sentiments = tokenizer.batch_decode(tokens)
  # 'positve'


logits = logits.squeeze(1)
print(f"==>> logits.shape: {logits.shape}")
print(f"==>> logits: {logits}")
# only take the logits of positive and negative
selected_logits = logits[:, [1465, 2841]] 
print(f"==>> selected_logits: {selected_logits}")

probs = F.softmax(selected_logits, dim=1)
print(f"==>> probs: {probs}")
#=> tensor([[0.9820, 0.0180]])

