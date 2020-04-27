

from sklearn.metrics import f1_score
import random

'''random baseline'''
out_label_ids = [1]*241910+[0]*25693
preds = []
for i in range(267603):
    prob = random.uniform(0, 1)
    if prob > 0.5:
        preds.append(0)
    else:
        preds.append(1)

# out_label_ids = [1]*1847+[0]*909
# preds = [1]*2756

f1 = f1_score(out_label_ids, preds, pos_label= 0, average='binary')
print(f1)
#
# f1 = f1_score(out_label_ids, preds, pos_label= 1, average='binary')
# print(f1)

# import torch
# from longformer.longformer import Longformer
# from transformers import RobertaTokenizer
#
# model = Longformer.from_pretrained('longformer-large-4096/')
# tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
# tokenizer.max_len = model.config.max_position_embeddings
#
# SAMPLE_TEXT = ' '.join(['Hello world! '] * 500)  # long input document
# SAMPLE_TEXT = f'{tokenizer.cls_token}{SAMPLE_TEXT}{tokenizer.eos_token}'
#
# input_ids = torch.tensor(tokenizer.encode(SAMPLE_TEXT)).unsqueeze(0)  # batch of size 1
#
# model = model.cuda()  # doesn't work on CPU
# input_ids = input_ids.cuda()
#
# # Attention mask values -- 0: no attention, 1: local attention, 2: global attention
# attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device) # initialize to local attention
# attention_mask[:, [1, 4, 21,]] =  2  # Set global attention based on the task. For example,
#                                      # classification: the <s> token
#                                      # QA: question tokenss
#
# output = model(input_ids, attention_mask=attention_mask)[0]
# print(output)
