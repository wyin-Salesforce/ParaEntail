
import json_lines
from collections import defaultdict
import operator
# from preprocess_hard import load_per_docs_file, load_DUC_doc
import codecs
import os

def load_DUC_doc(fil):
    readfile = codecs.open(fil, 'r', 'utf-8')

    doc_start = False
    doc = ''
    for line in readfile:
        if line.strip().find('<TEXT>') > -1:
            doc_start = True
            doc+=' '+line.strip().replace('<TEXT>', ' ')
            continue
        if line.strip().find('</TEXT>')>-1:
            doc+=' '+line.strip().replace('</TEXT>', ' ')
            doc_start = False
            break
        if doc_start:
            doc+=' '+line.strip()

    refined_doc = doc.replace('<P>', '').replace('</P>', '')
    readfile.close()
    return ' '.join(refined_doc.strip().split())

def load_per_docs_file(fil):
    print('fil:', fil)
    readfile = codecs.open(fil, 'r', 'utf-8')
    id2sum={}
    summary_start = False
    for line in readfile:
        if line.strip().startswith('<SUM'):
            doc_id = ''
            summary = ''
        if line.strip().startswith('DOCREF'):
            linestrip=line.strip()
            equi_pos = linestrip.find('=')
            doc_id = linestrip[equi_pos+2:-1]
        if line.strip().startswith('SUMMARIZER'):
            summary_start = True
            continue
        if line.strip().find('</SUM>') >-1:
            summary +=' '+line.strip().replace('</SUM>', ' ')
            # print('sum:', summary)
            summary_start = False
            id2sum[doc_id] = summary.strip()
        if summary_start:
            summary +=' '+line.strip()
    # print('size:', len(id2sum))
    return id2sum


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
    unvalid = 0
    for prefix in ['train', 'dev', 'test']:
        filename = path+prefix+'.txt'
        print('loading ANLI...', filename)
        with open(filename, 'r') as readfile:
            for line in readfile:
                parts = line.strip().split('\t')
                if len(parts) ==3:
                    premise = len(parts[1].split())
                    hypothesis = len(parts[2].split())
                    premise2times[premise]+=1
                    hypothesis2times[hypothesis]+=1
                    overal_size+=1
                else:
                    unvalid +=1

    print('overal_size:', overal_size, 'unvalid:', unvalid)
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

def count_length_DUC():
    trainfolder_namelist = ['d01a','d02a','d03a','d07b','d09b','d10b','d16c','d17c','d18c','d20d','d21d',
    'd23d','d25e','d26e','d29e','d33f','d35f','d36f','d38g','d40g','d42g','d46h','d47h','d48h','d49i',
    'd51i','d52i','d55k','d58k','d60k']

    test_folder_namelist = ['d04a','d05a','d06a','d08b','d11b','d12b','d13c','d14c','d15c','d19d','d22d',
    'd24d','d27e','d28e','d30e','d31f','d32f','d34f','d37g','d39g',
    'd41g','d43h','d44h','d45h','d50i','d53i','d54i','d56k','d57k','d59k']

    premise2times =  defaultdict(int)
    hypothesis2times =  defaultdict(int)

    for folder_namelist in [trainfolder_namelist, test_folder_namelist]:
        size = 0
        for foldername in folder_namelist:
            last_char = foldername[-1]
            subfolder = foldername+last_char
            docsfolder = 'docs'
            perdoc_file ='/export/home/Dataset/para_entail_datasets/DUC/DUC_data/data/duc01/data/training/'+foldername+'/'+subfolder+'/perdocs'
            id2sum = load_per_docs_file(perdoc_file)

            id2doc = {}
            path_to_stories = '/export/home/Dataset/para_entail_datasets/DUC/DUC_data/data/duc01/data/training/'+foldername+'/docs/'
            story_filenames_list = os.listdir(path_to_stories)
            for story_filename in story_filenames_list:
                path_to_story = os.path.join(path_to_stories, story_filename)
                if os.path.isfile(path_to_story):
                    doc = load_DUC_doc(path_to_story)
                    id2doc[story_filename] = doc


            for id, doc in id2doc.items():
                # print(id, '\n', doc, '\n', id2sum.get(id))
                doc_str = doc

                summ = id2sum.get(id)
                if summ is None or len(doc_str.strip()) == 0:
                    continue

                premise2times[len(doc_str.split())]+=1
                hypothesis2times[len(summ.split())]+=1

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
    # count_length_SQUAD()
    count_length_DUC()
