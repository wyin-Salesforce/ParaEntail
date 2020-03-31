from transformers import AutoModelWithLMHead, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
model = AutoModelWithLMHead.from_pretrained("distilbert-base-cased")

sequence = f"Distilled models are smaller than the models they mimic. Using them instead of the large versions would help {tokenizer.mask_token} our carbon footprint. Distilled models are smaller than the models they mimic. Using them instead of the large versions would help Distilled models are smaller than the models they mimic. Using them instead of the large versions would help Distilled models are smaller than the models they mimic. Using them instead of the large versions would help Distilled models are smaller than the models they mimic. Using them instead of the large versions would help Distilled models are smaller than the models they mimic. Using them instead of the large versions would help Distilled models are smaller than the models they mimic. Using them instead of the large versions would help"

input = tokenizer.encode(sequence, return_tensors="pt")
mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]

token_logits = model(input)[0]
print(token_logits)
