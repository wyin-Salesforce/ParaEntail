import codecs
import json
from sklearn.metrics import f1_score
import random
random.seed(42)


readfile = codecs.open('/export/home/Dataset/para_entail_datasets/test.json', 'r', 'utf-8')

data = json.load(readfile)
gold_labels = []
pred_labels = []
print('len of data:', len(data))
for dic in data:
    gold_label = 1 if dic.get('label') == 'entailment' else 0
    pred_label = 1 if random.uniform(0, 1)>0.5 else 0
    gold_labels.append(gold_label)
    pred_labels.append(pred_label)


f1 = f1_score(gold_labels, pred_labels, pos_label= 1, average='binary')
print(f1)
