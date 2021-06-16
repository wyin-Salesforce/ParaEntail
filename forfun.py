
from load_data import load_DocNLI

def examples_2_dic(examples):
    dic = set()
    for ex in examples:
        dic.add((ex.text_a, ex.text_b))
    return dic
train_examples = load_DocNLI('train', hypo_only=False)
train_set = examples_2_dic(train_examples)
dev_examples = load_DocNLI('test', hypo_only=False)
dev_set = examples_2_dic(dev_examples)

joint_set = train_set.intersection(dev_set)
print(len(joint_set))
