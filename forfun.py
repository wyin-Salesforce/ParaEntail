

# from sklearn.metrics import f1_score
# import random
#
# '''random baseline'''
# # out_label_ids = [1]*241910+[0]*25693
# # preds = []
# # for i in range(267603):
#
# out_label_ids = [1]*211628+[0]*22042
# preds = []
# for i in range(233670):
#     prob = random.uniform(0, 1)
#     if prob > 0.5:
#         preds.append(0)
#     else:
#         preds.append(1)
#
#
#
# f1 = f1_score(out_label_ids, preds, pos_label= 0, average='binary')
# print(f1)

from load_data import load_harsh_data
from collections import defaultdict
examples_all = []
examples = load_harsh_data('train', hypo_only=False)
examples_all+=examples
examples_dev = load_harsh_data('dev', hypo_only=False)
examples_all+=examples_dev
examples_test = load_harsh_data('test', hypo_only=False)
examples_all+=examples_test

print('example size:', len(examples_all))


premise_len2size = defaultdict(int)
hypothesis_len2size = defaultdict(int)

count = 0
for ex in examples_all:
    premise_len = len(ex.text_a.split())
    idd = int(premise_len/50)
    premise_len2size[idd] = premise_len2size.get(idd, 0)+1

    hypothesis_len = len(ex.text_b.split())
    idd = int(hypothesis_len/50)
    hypothesis_len2size[idd] = hypothesis_len2size.get(idd, 0)+1
    count+=1
    if count % 1000 ==0:
        print('count:', count)


print('premise_len2size:', premise_len2size)
print('hypothesis_len2size:', hypothesis_len2size)
