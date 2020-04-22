

from sklearn.metrics import f1_score
import random

'''random baseline'''
# out_label_ids = [1]*1847+[0]*909
# preds = []
# for i in range(2756):
#     prob = random.uniform(0, 1)
#     if prob > 0.5:
#         preds.append(0)
#     else:
#         preds.append(1)

out_label_ids = [1]*1847+[0]*909
preds = [1]*2756

f1 = f1_score(out_label_ids, preds, pos_label= 0, average='binary')
print(f1)
