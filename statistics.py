
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

    print('overal_size:', overal_size)
    main_size = int(overal_size*0.95)
    print('main_size:', main_size)


    for origin_dict in [premise2times, hypothesis2times]:
        premise2times_sorted = dict(sorted(origin_dict.items(), key=operator.itemgetter(1),reverse=True))
        # print('premise2times_sorted:', premise2times_sorted)
        value_sum = 0
        remain_premise_dict = {}
        for length, times in premise2times_sorted.items():
            value_sum+=times
            if value_sum > main_size:
                min_premise = length
                break
            else:
                remain_premise_dict[length] = times
        remain_premise_dict = dict(sorted(remain_premise_dict.items(), key=operator.itemgetter(0),reverse=True))
        print('remain_premise_dict:', remain_premise_dict)

def count_length_SQUAD():

    neg_size = 0
    path = '/export/home/Dataset/para_entail_datasets/SQUAD/'
    premise2times =  defaultdict(int)
    hypothesis2times =  defaultdict(int)
    overal_size = 0

    for prefix in ['train', 'dev', 'test']:
        filename = path+prefix+'.txt'
        print('loading ANLI...', filename)
        with open(filename, 'r') as readfile:
            for line in readfile:
                parts = line.strip().split('\t')
                premise = len(parts[1].split())
                hypothesis = len(parts[2].split())
                premise2times[premise]+=1
                hypothesis2times[hypothesis]+=1
                overal_size+=1

    print('overal_size:', overal_size)
    main_size = int(overal_size*0.95)
    print('main_size:', main_size)


    for origin_dict in [premise2times, hypothesis2times]:
        premise2times_sorted = dict(sorted(origin_dict.items(), key=operator.itemgetter(1),reverse=True))
        # print('premise2times_sorted:', premise2times_sorted)
        value_sum = 0
        remain_premise_dict = {}
        for length, times in premise2times_sorted.items():
            value_sum+=times
            if value_sum > main_size:
                min_premise = length
                break
            else:
                remain_premise_dict[length] = times
        remain_premise_dict = dict(sorted(remain_premise_dict.items(), key=operator.itemgetter(0),reverse=True))
        print('remain_premise_dict:', remain_premise_dict)



if __name__ == "__main__":
    # count_length_ANLI()
    count_length_SQUAD()
