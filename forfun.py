from transformers import AutoModelWithLMHead, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
model = AutoModelWithLMHead.from_pretrained("distilbert-base-cased")

prior_str = 'Distilled models are smaller than the models they mimic. Using them instead of the large versions would help Distilled models are smaller than the models they mimic. Using them instead of the large versions would help Distilled models are smaller than the models they mimic. Using them instead of the large versions would help Distilled models are smaller than the models they mimic. Using them instead of the large versions would help Distilled models are smaller than the models they mimic. Using them instead of the large versions would help Distilled models are smaller than the models they mimic. Using them instead of the large versions would help Distilled models are smaller than the models they mimic. Using them instead of the large versions would help Distilled models are smaller than the models they mimic. Using them instead of the large versions would help Distilled models are smaller than the models they mimic. Using them instead of the large versions would help Distilled models are smaller than the models they mimic. Using them instead of the large versions would help Distilled models are smaller than the models they mimic. Using them instead of the large versions would help Distilled models are smaller than the models they mimic. Using them instead of the large versions would help Distilled models are smaller than the models they mimic. Using them instead of the large versions would help Distilled models are smaller than the models they mimic. Using them instead of the large versions would help Distilled models are smaller than the models they mimic. Using them instead of the large versions would help Distilled models are smaller than the models they mimic. Using them instead of the large versions would help Distilled models are smaller than the models they mimic. Using them instead of the large versions would help Distilled models are smaller than the models they mimic. Using them instead of the large versions would help Distilled models are smaller than the models they mimic. Using them instead of the large versions would help '

after_str = ' our carbon footprint. Distilled models are smaller than the models they mimic. Using them instead of the large versions would help Distilled models are smaller than the models they mimic. Using them instead of the large versions would help Distilled models are smaller than the models they mimic. Using them instead of the large versions would help Distilled models are smaller than the models they mimic. Using them instead of the large versions would help Distilled models are smaller than the models they mimic. Using them instead of the large versions would help Distilled models are smaller than the models they mimic. Using them instead of the large versions would help'

prior_str = ' '.join(prior_str.split()[:400])
print('prior_str len:', len(prior_str.split()))
print('after_str len:', len(after_str.split()))
sequence = f"{tokenizer.mask_token}"
sequence = prior_str + sequence + after_str

input = tokenizer.encode(sequence, return_tensors="pt", max_length=512)
mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]

print('input len:', input.size())
token_logits = model(input)[0]
print(token_logits)
