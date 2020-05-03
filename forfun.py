

from sklearn.metrics import f1_score
import random

'''random baseline'''
# out_label_ids = [1]*241910+[0]*25693
# preds = []
# for i in range(267603):

out_label_ids = [1]*87518+[0]*23807
preds = []
for i in range(111425):
    prob = random.uniform(0, 1)
    if prob > 0.5:
        preds.append(0)
    else:
        preds.append(1)



f1 = f1_score(out_label_ids, preds, pos_label= 0, average='binary')
print(f1)
