

from sklearn.metrics import f1_score
import random

out_label_ids = [1]*102882+[0]*20975
preds = []
for i in range(123857):
    prob = random.uniform(0, 1)
    if prob > 0.5:
        preds.append(0)
    else:
        preds.append(1)



f1 = f1_score(out_label_ids, preds, pos_label= 0, average='binary')
print(f1)
