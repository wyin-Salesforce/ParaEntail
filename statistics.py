
import json_lines

def count_length_ANLI():
    folders = ['R1', 'R2', 'R3']
    examples = []
    guid_id = 0
    pos_size = 0
    neg_size = 0
    path = '/export/home/Dataset/para_entail_datasets/ANLI/anli_v0.1/'
    premise_min = 1000
    premise_max = 0
    hypothesis_min = 1000
    hypothesis_max = 0
    for folder in folders:
        for prefix in ['train', 'dev', 'test']:
            filename = path+folder+'/'+prefix+'.jsonl'
            print('loading ANLI...', filename)
            with open(filename, 'r') as f:
                for line in json_lines.reader(f):
                    guid_id+=1
                    premise = len(line.get('context').split())
                    hypothesis = len(line.get('hypothesis').split())

                    if premise>premise_max:
                        premise_max = premise
                    if premise < premise_min:
                        premise_min = premise

                    if hypothesis > hypothesis_max:
                        hypothesis_max = hypothesis
                    if hypothesis < hypothesis_min:
                        hypothesis_min = hypothesis

    print(premise_min, premise_max, hypothesis_min, hypothesis_max)


if __name__ == "__main__":
    count_length_ANLI()
