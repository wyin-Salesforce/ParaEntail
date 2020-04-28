

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
from collections import OrderedDict
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
    idd = int(premise_len/100)
    premise_len2size[idd] = premise_len2size.get(idd, 0)+1

    hypothesis_len = len(ex.text_b.split())
    idd = int(hypothesis_len/50)
    if idd > 4:
        idd = 4
    hypothesis_len2size[idd] = hypothesis_len2size.get(idd, 0)+1
    count+=1
    if count % 1000 ==0:
        print('count:', count)


sorted_pre = OrderedDict(sorted(premise_len2size.items()))
sorted_hyp = OrderedDict(sorted(hypothesis_len2size.items()))

print('sorted_pre:', sorted_pre)
print(list(sorted_pre.values()))
print('sorted_hyp:', sorted_hyp)
print(list(sorted_hyp.values()))

'''
[542161, 850081, 152585, 9733, 1821, 919, 385, 378, 162, 288, 23, 9, 30, 3]
'''
