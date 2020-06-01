
import json_lines
from collections import defaultdict
import operator

def count_length_ANLI():
    folders = ['R1', 'R2', 'R3']
    examples = []
    guid_id = 0
    pos_size = 0
    neg_size = 0
    path = '/export/home/Dataset/para_entail_datasets/ANLI/anli_v0.1/'
    premise2times =  defaultdict(int)
    hypothesis2times =  defaultdict(int)
    overal_size = 0
    for folder in folders:
        for prefix in ['train', 'dev', 'test']:
            filename = path+folder+'/'+prefix+'.jsonl'
            print('loading ANLI...', filename)
            with open(filename, 'r') as f:
                for line in json_lines.reader(f):
                    guid_id+=1
                    premise = len(line.get('context').split())
                    hypothesis = len(line.get('hypothesis').split())
                    premise2times[premise]+=1
                    hypothesis2times[hypothesis]+=1
                    overal_size+=1

    main_size = int(overal_size*0.9)
    premise2times_sorted = dict(sorted(premise2times.items(), key=operator.itemgetter(1),reverse=True))

    value_sum = 0
    max_premise = 0
    min_premise = 10000000
    i=0
    for key, value in premise2times_sorted.items():
        if i == 0:
            max_premise = value
        value_sum+=value
        if value_sum > main_size:
            min_premise = value

    print(max_premise, min_premise)





    # print(premise_min, premise_max, hypothesis_min, hypothesis_max)


if __name__ == "__main__":
    count_length_ANLI()
