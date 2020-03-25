
import codecs
import os
import nltk
from nltk import pos_tag, ne_chunk
from nltk.tokenize import word_tokenize
from nltk.chunk import conlltags2tree, tree2conlltags
import random

import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm

def load_CNN_DailyMail():
    trainfile = codecs.open('/export/home/Dataset/CNN-DailyMail-Summarization/split/train_tokenized.txt', 'r', 'utf-8')
    for line in trainfile:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            train_src.write(parts[0].strip()+'\n')
            train_trg.write(parts[1].strip()+'\n')
    trainfile.close()

    writeval = codecs.open('/export/home/Dataset/CNN-DailyMail-Summarization/split/val_tokenized.txt', 'r', 'utf-8')

    writetest = codecs.open('/export/home/Dataset/CNN-DailyMail-Summarization/split/test_tokenized.txt', 'r', 'utf-8')



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
    print('size:', len(id2sum))
    return id2sum

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
    readfile.close()
    return doc.strip()

def appearance_of_str(mom_str, baby_str):
    poslist = []

    prior_len = 0
    while len(mom_str) > 0:
        pos = mom_str.find(baby_str)
        if pos > -1:
            start = pos
            end = start + len(baby_str)
            poslist.append((start+prior_len, end+prior_len))
            prior_len += end
            mom_str = mom_str[end:]

    for pos in poslist:
        print(mom_str[pos[0]:pos[1]])

    return poslist


def word_change(doc_str, sum_str):
    '''swap entity'''
    doc_ent_dict = NER(doc_str)
    sum_ent_dict = NER(sum_str)

    new_sum = sum_str
    for nerlabel, valuelist  in sum_ent_dict.items():
        if len(valuelist) == 1:
            doc_valuelist = doc_ent_dict.get(nerlabel)
            if doc_valuelist is None:
                continue
            else:
                #fine one from doc
                doc_ent = random.choice(doc_valuelist)
                if doc_ent != valuelist[0]:
                    new_sum = new_sum.replace(valuelist[0], doc_ent)
                else:
                    break
        else:
            #swap inside
            entities_to_swap = random.sample(valuelist, 2)





def NER(input):

    nlp = en_core_web_sm.load()
    # doc = nlp('European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices')
    doc = nlp(input)

    nerlabel2entitylist = {}
    for X in doc.ents:
        entlist = nerlabel2entitylist.get(X.label_)
        if entlist is None:
            entlist = []
        entlist.append(X.text)
        nerlabel2entitylist[X.label_] = entlist

    return nerlabel2entitylist

def generate_negative_summaries(doc_str, sum_str):
    return


def load_DUC():
    #DUC2001
    trainfolder_namelist = ['d01a','d02a','d03a','d07b','d09b','d10b','d16c','d17c','d18c','d20d','d21d',
    'd23d','d25e','d26e','d29e','d33f','d35f','d36f','d38g','d40g','d42g','d46h','d47h','d48h','d49i',
    'd51i','d52i','d55k','d58k','d60k']

    for foldername in trainfolder_namelist:
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

        print(id2doc.keys())
        print(id2sum.keys())
        assert len(id2doc) ==  len(id2sum)

        for id, doc in id2doc.items():
            print(id, '\n', doc, '\n', id2sum.get(id))
            exit(0)








if __name__ == "__main__":
    # load_per_docs_file('/export/home/Dataset/para_entail_datasets/DUC/DUC_data/data/duc01/data/training/d49i/d49ii/perdocs')
    # load_DUC()
    # NER('European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices.')
    appearance_of_str('why we do there without why you come why why .')
