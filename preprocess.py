
import codecs
import os
import nltk
# from nltk import pos_tag, ne_chunk
# from nltk.tokenize import word_tokenize
# from nltk.chunk import conlltags2tree, tree2conlltags
import random
import torch
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
from collections import defaultdict
from transformers import AutoModelWithLMHead, AutoTokenizer
# from transformers import AutoModelWithLMHead, AutoTokenizer
from transformers import pipeline
import numpy as np
import xmltodict
import json_lines
import json
import csv
import pandas as pd
from fastprogress.fastprogress import progress_bar
import aiohttp
import asyncio
from bs4 import BeautifulSoup
import csv
from fastprogress.fastprogress import progress_bar
import os
import pandas as pd
from readability import Document
from sys import argv


seed = 400
random.seed(seed)
np.random.seed(seed)
device = torch.device("cuda")


def load_CNN_DailyMail():
    mask_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
    mask_model = AutoModelWithLMHead.from_pretrained("distilbert-base-cased")
    mask_model.to(device)

    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    gpt2_model = AutoModelWithLMHead.from_pretrained("gpt2")
    gpt2_model.to(device)

    file_prefix = ['val']#['train', 'val', 'test']
    for fil_prefix in file_prefix:
        readfil = '/export/home/Dataset/CNN-DailyMail-Summarization/split/'+fil_prefix+'_tokenized.txt'
        writefil = '/export/home/Dataset/para_entail_datasets/CNN_DailyMail/'+fil_prefix+'_in_entail.txt'
        readfile = codecs.open(readfil, 'r', 'utf-8')
        writefile = codecs.open(writefil, 'w', 'utf-8')
        size = 0
        skip_overlong_sum_size = 0
        prior_unrelated_doc = "Donald John Trump is the 45th and current president of the United States. Before entering politics, he was a businessman and television personality. Trump was born and raised in Queens, a borough of New York City, and received a bachelor's degree in economics from the Wharton School."
        for line in readfile:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                # size+=1
                # if size <= 3725:
                #     continue
                doc_str = parts[0].strip()
                writefile.write('document>>' +'\t'+doc_str+'\n')
                sum_str = parts[1].strip()
                if len(sum_str.split()) > 200:
                    skip_overlong_sum_size+=1
                    continue
                writefile.write('positive>>' +'\t'+ sum_str+'\n')
                neg_sum_list, neg_sum_namelist = generate_negative_summaries(prior_unrelated_doc, doc_str, sum_str, mask_tokenizer, mask_model, gpt2_tokenizer, gpt2_model)
                prior_unrelated_doc = doc_str
                for id, neg_sum in enumerate(neg_sum_list):
                    writefile.write('negative>>' +'\t'+neg_sum_namelist[id]+'>>\t'+neg_sum+'\n')
                writefile.write('\n')
                size+=1
                if size % 500 == 0:
                    print(fil_prefix, ' doc size:', size)
        readfile.close()
        writefile.close()

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

def appearance_of_str(mom_str, baby_str):
    poslist = {}
    origin_mom_str = mom_str

    prior_len = 0
    while len(mom_str) > 0:
        # print('mom_str:', mom_str)
        pos = mom_str.find(baby_str)
        # print('pos:', pos)
        if pos > -1:
            start = pos
            end = start + len(baby_str)
            # print('start:', start, 'end:', end, 'prior_len:', prior_len)
            # poslist.append((start+prior_len, end+prior_len))
            poslist[start+prior_len] = end+prior_len
            prior_len += end
            mom_str = mom_str[end:]
        else:
            break

    for pos_key, pos_value in poslist.items():
        assert origin_mom_str[pos_key:pos_value] == baby_str

    return poslist


def swap_entities(doc_str, sum_str):
    '''swap entity'''
    doc_ent_dict = NER(doc_str)
    sum_ent_dict = NER(sum_str)
    # print('sum_ent_dict:', sum_ent_dict)


    negative_sum_list = []

    # new_sum = sum_str
    for nerlabel, valuelist  in sum_ent_dict.items():
        '''for each NER type, e.g., person, org'''
        if len(valuelist) == 1:
            '''if summary has only one entity, swap with doc'''
            doc_valuelist = doc_ent_dict.get(nerlabel)
            if doc_valuelist is None:
                continue
            else:
                #fine one from doc
                doc_ent = random.choice(doc_valuelist)
                if doc_ent != valuelist[0]:
                    new_sum = sum_str.replace(valuelist[0], doc_ent)
                    negative_sum_list.append(new_sum)
                else:
                    continue
        else:
            '''if summary has more than two, swap within summary'''
            entities_to_swap = random.sample(valuelist, 2)
            '''avoid swap in A and B'''
            if sum_str.find(entities_to_swap[0]+' and '+entities_to_swap[1]) > -1 or sum_str.find(entities_to_swap[1]+' and '+entities_to_swap[0]) > -1:
                continue
            pos_dict_0 = appearance_of_str(sum_str, entities_to_swap[0])
            # print('pos_dict_0:', pos_dict_0)
            pos_dict_1 = appearance_of_str(sum_str, entities_to_swap[1])
            # print('pos_dict_1:', pos_dict_1)
            '''combine two dict'''
            pos_dict_combine = {**pos_dict_0, **pos_dict_1}
            # print('pos_dict_combine:', pos_dict_combine)

            prior_str = ''
            prior_end = 0
            for i in sorted (pos_dict_combine.keys()):
                # print('i:',i)
                prior_str += sum_str[prior_end:i]
                end_0 = pos_dict_0.get(i)
                end_1 = pos_dict_1.get(i)
                # print('end_0:', end_0, 'end_1:', end_1)
                if end_0 is None and end_1 is None:
                    print('error')
                    exit(0)
                elif end_0 is not None:
                    '''entity A, replace by entity B'''
                    prior_str += entities_to_swap[1]
                    prior_end = end_0
                else:
                    prior_str += entities_to_swap[0]
                    prior_end = end_1
            prior_str += sum_str[prior_end:]

            if prior_str.strip() != sum_str.strip():
                negative_sum_list.append(prior_str)

            # print('origin sum:', sum_str)
            # print('new sum:', prior_str)
            # exit(0)

    return negative_sum_list



def swap_pronouns(sum_str):
    '''dont plan to use'''
    pronouns_cands = {'he': ['she'],
    'she': ['he'],
    'his': ['her'],
    'her': ['his']
    }

def shuffle_words_same_POStags(sum_str, prob):
    preferred_POStags = set(['VERB', 'NOUN', 'PROPN', 'NUM'])
    nlp = en_core_web_sm.load()
    doc = nlp(sum_str)
    pos2words = defaultdict(list)
    for token in doc:
        # print(token.text, '>>', token.pos_)
        pos2words[token.pos_].append(token)
    new_word_list = []
    for token in doc:
        '''for each token, replace it by prob'''
        if token.pos_ not in preferred_POStags:
            new_word_list.append(token.text)
        else:
            word_set = set(pos2words.get(token.pos_))
            if len(word_set) ==  1:
                new_word_list.append(token.text)
                continue
            else:
                word_set.discard(token)
                assert len(word_set) >=1
                rand_prob = random.uniform(0, 1)
                if rand_prob > prob:
                    '''do not replace'''
                    new_word_list.append(token.text)
                    continue
                else:
                    replace_word = random.choice(list(word_set))
                    new_word_list.append(replace_word.text)
    # print('old:', sum_str)
    # print('new:', ' '.join(new_word_list))
    return [' '.join(new_word_list)]

def random_remove_words(sum_str, drop):
    tokens = sum_str.strip().split()
    remove_size = int(len(tokens) * drop)
    removed_indices = set(random.sample(list(range(len(tokens))),remove_size))

    new_tokens = []
    for idd, word in enumerate(tokens):
        if idd not in removed_indices:
            new_tokens.append(word)
        else:
            continue

    return [' '.join(new_tokens)]

def random_replace_words(sum_str, drop, tokenizer, model):

    input_wordlist = sum_str.strip().split()
    sum_len = len(input_wordlist)
    replace_size = int(sum_len*drop)#0.3

    prior_sum = input_wordlist
    for i in range(replace_size):
        prior_len = len(prior_sum)
        pos = random.randrange(prior_len-1)
        '''to avoid error: Tried to access index 512 out of table with 511 rows'''
        pos = min(pos, 200)
        sequence = ' '.join(prior_sum[:pos])+' '+ f"{tokenizer.mask_token}" + ' '+ ' '.join(prior_sum[pos+1:])

        '''set max_length = 512 is necessary, but does not work gor gpt2'''
        input = tokenizer.encode(sequence, return_tensors="pt", max_length=512)
        mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]
        input = input.to(device)

        token_logits = model(input)[0]
        mask_token_logits = token_logits[0, mask_token_index, :]

        top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
        # print(top_5_tokens)

        prior_sum = sequence.replace(tokenizer.mask_token, tokenizer.decode([top_5_tokens[0]])).split()
        # print(' '.join(prior_sum))

    return [' '.join(prior_sum)]

def append_unrelated_sents(sum_str, prior_unrelated_doc):
    # nlp = spacy.load('en_core_web_sm')
    # text = "Donald John Trump is the 45th and current president of the United States. Before entering politics, he was a businessman and television personality. Trump was born and raised in Queens, a borough of New York City, and received a bachelor's degree in economics from the Wharton School."
    # text_sentences = nlp(text)
    # for sentence in text_sentences.sents:

    nlp = en_core_web_sm.load()
    text_sentences = nlp(sum_str)
    sum_sents = []
    for sentence in text_sentences.sents:
        sum_sents.append(sentence.text)

    # print('append_unrelated_sents.prior_unrelated_doc:', prior_unrelated_doc)
    doc_sentences = nlp(prior_unrelated_doc)
    doc_sents = []
    for sentence in doc_sentences.sents:
        doc_sents.append(sentence.text)
    if len(doc_sents) == 0:
        print('append_unrelated_sents.prior_unrelated_doc:', prior_unrelated_doc)
        exit(0)
    random_sent_from_doc = random.choice(doc_sents)
    '''put the unrelated sent at the position 1'''
    new_sum_sents = sum_sents[:1]+[random_sent_from_doc]+sum_sents[1:]


    return [' '.join(new_sum_sents)]

def GPT2_generate(sum_str, tokenizer, model):
    # print('sum_str:', sum_str)
    input_wordlist = sum_str.split()
    input_len = len(input_wordlist)
    max_len = input_len+20

    keep_lengths = [int(input_len*0.3), int(input_len*0.6), int(input_len*0.9)]
    new_seqs = []
    for leng in keep_lengths:

        sequence = ' '.join(input_wordlist[:leng])#f"Hugging Face is based in DUMBO, New York City, and is"
        # print('sequence:', sequence)
        input = tokenizer.encode(sequence, return_tensors="pt")
        input = input.to(device)
        # print('input:', input)
        generated = model.generate(input, max_length=max_len)

        resulting_string = ' '.join(tokenizer.decode(generated.tolist()[0]).strip().split())
        # print('resulting_string:', resulting_string)
        new_seq = resulting_string[:resulting_string.rfind('.')+1]
        # print(resulting_string.rfind('.'), len(sum_str))
        if len(new_seq.split()) > leng:
            new_seqs.append(new_seq)
    # print(new_seqs)

    return new_seqs


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

def generate_negative_summaries(prior_unrelated_doc, doc_str, sum_str, mask_tokenizer, mask_model, gpt2_tokenizer, gpt2_model):
    '''entity-level noise'''
    entity_cand_list = swap_entities(doc_str, sum_str)
    entity_cand_list_names = ['#SwapEnt#'] * len(entity_cand_list)
    # swap_pronouns(doc_str, sum_str)
    '''word-level noise'''
    shuffle_word_list = shuffle_words_same_POStags(sum_str, 0.5)
    shuffle_word_list_names = ['#ShuffleWord#'] * len(shuffle_word_list)

    missing_word_list = random_remove_words(sum_str, 0.2)
    missing_word_list_names = ['#RemoveWord#'] * len(missing_word_list)

    bert_mask_list = random_replace_words(sum_str, 0.2, mask_tokenizer, mask_model)
    bert_mask_list_names = ['#ReplaceWord#'] * len(bert_mask_list)

    '''sentence-level noise'''

    insert_unrelated_sents = append_unrelated_sents(sum_str, prior_unrelated_doc)
    insert_unrelated_sents_names = ['#InsertUnrelatedSent#'] * len(insert_unrelated_sents)

    bert_generate_list = GPT2_generate(sum_str, gpt2_tokenizer, gpt2_model)
    bert_generate_list_names = ['#GPT2generate#'] * len(bert_generate_list)

    cand_list= entity_cand_list + shuffle_word_list + missing_word_list + bert_mask_list + insert_unrelated_sents+bert_generate_list
    name_list = entity_cand_list_names + shuffle_word_list_names+missing_word_list_names + bert_mask_list_names+ insert_unrelated_sents_names + bert_generate_list_names

    return cand_list, name_list

def load_DUC_train():
    #DUC2001
    trainfolder_namelist = ['d01a','d02a','d03a','d07b','d09b','d10b','d16c','d17c','d18c','d20d','d21d',
    'd23d','d25e','d26e','d29e','d33f','d35f','d36f','d38g','d40g','d42g','d46h','d47h','d48h','d49i',
    'd51i','d52i','d55k','d58k','d60k']

    writefile = codecs.open('/export/home/Dataset/para_entail_datasets/DUC/train_in_entail.txt', 'w', 'utf-8')
    mask_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
    mask_model = AutoModelWithLMHead.from_pretrained("distilbert-base-cased")
    mask_model.to(device)

    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    gpt2_model = AutoModelWithLMHead.from_pretrained("gpt2")
    gpt2_model.to(device)

    size = 0
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

        # print(id2doc.keys())
        # print(id2sum.keys())
        # assert len(id2doc) ==  len(id2sum)
        prior_unrelated_doc = "Donald John Trump is the 45th and current president of the United States. Before entering politics, he was a businessman and television personality. Trump was born and raised in Queens, a borough of New York City, and received a bachelor's degree in economics from the Wharton School."
        for id, doc in id2doc.items():
            # print(id, '\n', doc, '\n', id2sum.get(id))
            doc_str = doc

            summ = id2sum.get(id)
            if summ is None or len(doc_str.strip()) == 0:
                print('missing:', foldername, id)
                continue

            writefile.write('document>>' +'\t'+doc_str+'\n')
            sum_str = ' '.join(summ.strip().split())

            # writefile.write('positive' +'\t'+doc_str + '\t' + sum_str+'\n')
            writefile.write('positive>>' +'\t'+sum_str+'\n')
            # print('load_DUC_train.prior_unrelated_doc:', prior_unrelated_doc)
            neg_sum_list, neg_sum_namelist = generate_negative_summaries(prior_unrelated_doc, doc_str, sum_str, mask_tokenizer, mask_model, gpt2_tokenizer, gpt2_model)
            prior_unrelated_doc = doc_str
            # print('load_DUC_train.prior_unrelated_doc.update:', prior_unrelated_doc)
            for id, neg_sum in enumerate(neg_sum_list):
                writefile.write('negative>>' +'\t'+neg_sum_namelist[id]+'>>\t'+neg_sum+'\n')
            writefile.write('\n')
            # writefile.close()
            # exit(0)
            # for neg_sum in neg_sum_list:
            #     writefile.write('negative' +'\t'+doc_str + '\t' + neg_sum+'\n')
            size+=1
            if size % 10 == 0:
                print('doc size:', size)
    writefile.close()


def load_DUC_test():
    #DUC2001
    # trainfolder_namelist = ['d01a','d02a','d03a','d07b','d09b','d10b','d16c','d17c','d18c','d20d','d21d',
    # 'd23d','d25e','d26e','d29e','d33f','d35f','d36f','d38g','d40g','d42g','d46h','d47h','d48h','d49i',
    # 'd51i','d52i','d55k','d58k','d60k']

    test_folder_namelist = ['d04a','d05a','d06a','d08b','d11b','d12b','d13c','d14c','d15c','d19d','d22d',
    'd24d','d27e','d28e','d30e','d31f','d32f','d34f','d37g','d39g',
    'd41g','d43h','d44h','d45h','d50i','d53i','d54i','d56k','d57k','d59k']

    writefile = codecs.open('/export/home/Dataset/para_entail_datasets/DUC/test_in_entail.txt', 'w', 'utf-8')
    mask_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
    mask_model = AutoModelWithLMHead.from_pretrained("distilbert-base-cased")
    mask_model.to(device)

    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    gpt2_model = AutoModelWithLMHead.from_pretrained("gpt2")
    gpt2_model.to(device)

    '''test doc has multiple summary'''
    duplicate_sum_pathstring = '/export/home/Dataset/para_entail_datasets/DUC/DUC_data/data/duc01/data/test/duplicate.summaries'
    duplicate_sum_path = os.listdir(duplicate_sum_pathstring)
    folder_2_multiple = defaultdict(list)
    for foldername in duplicate_sum_path:
        path_to_folder = os.path.join(duplicate_sum_pathstring, foldername)
        if os.path.isdir(path_to_folder):
            folder_2_multiple[foldername[:4]].append(foldername)

    # print('folder_2_multiple:', folder_2_multiple)
    # exit(0)
    size = 0
    for foldername in test_folder_namelist:
        last_char = foldername[-1]
        subfolder = foldername+last_char
        docsfolder = 'docs'

        id2sumlist = defaultdict(list)
        # /export/home/Dataset/para_entail_datasets/DUC/DUC_data/data/duc01/data/test/original.summaries/d56kk
        perdoc_file ='/export/home/Dataset/para_entail_datasets/DUC/DUC_data/data/duc01/data/test/original.summaries'+'/'+subfolder+'/perdocs'
        id2sum = load_per_docs_file(perdoc_file)
        for idd, sum_i in id2sum.items():
            id2sumlist[idd].append(sum_i)

        '''load duplicate summary'''
        for sumfolder in folder_2_multiple.get(foldername):
            #/export/home/Dataset/para_entail_datasets/DUC/DUC_data/data/duc01/data/test/duplicate.summaries/d59kg
            perdoc_file ='/export/home/Dataset/para_entail_datasets/DUC/DUC_data/data/duc01/data/test/duplicate.summaries'+'/'+sumfolder+'/perdocs'
            id2sum_i = load_per_docs_file(perdoc_file)
            for idd, sum_i in id2sum_i.items():
                id2sumlist[idd].append(sum_i)

        id2doc = {}
        path_to_stories = '/export/home/Dataset/para_entail_datasets/DUC/DUC_data/data/duc01/data/test/docs/'+foldername+'/'
        story_filenames_list = os.listdir(path_to_stories)
        for story_filename in story_filenames_list:
            path_to_story = os.path.join(path_to_stories, story_filename)
            if os.path.isfile(path_to_story):
                doc = load_DUC_doc(path_to_story)
                id2doc[story_filename] = doc

        # print(id2doc.keys())
        # print(id2sum.keys())
        # assert len(id2doc) ==  len(id2sum)
        prior_unrelated_doc = "Donald John Trump is the 45th and current president of the United States. Before entering politics, he was a businessman and television personality. Trump was born and raised in Queens, a borough of New York City, and received a bachelor's degree in economics from the Wharton School."
        for id, doc in id2doc.items():
            # print(id, '\n', doc, '\n', id2sum.get(id))
            doc_str = doc#' '.join(doc.strip().split())

            summ_list = id2sumlist.get(id)
            if summ_list is None or len(doc_str.strip()) == 0:
                print('missing:', foldername, id)
                continue

            writefile.write('document>>' +'\t'+doc_str+'\n')
            for summm in summ_list:
                sum_str = ' '.join(summm.strip().split())
                # writefile.write('positive' +'\t'+doc_str + '\t' + sum_str+'\n')
                writefile.write('positive>>' +'\t'+sum_str+'\n')

            '''to save time, we only use the first summary to generate negative ones'''
            sum_str = ' '.join(summ_list[0].strip().split())
            neg_sum_list, neg_sum_namelist = generate_negative_summaries(prior_unrelated_doc, doc_str, sum_str, mask_tokenizer, mask_model, gpt2_tokenizer, gpt2_model)
            prior_unrelated_doc = doc_str
            for id, neg_sum in enumerate(neg_sum_list):
                writefile.write('negative>>' +'\t'+neg_sum_namelist[id]+'>>\t'+neg_sum+'\n')
            writefile.write('\n')
            # for neg_sum in neg_sum_list:
            #     writefile.write('negative' +'\t'+doc_str + '\t' + neg_sum+'\n')
            size+=1
            if size % 10 == 0:
                print('doc size:', size)
    writefile.close()


def load_MCTest(filenames, prefix):
    path = '/export/home/Dataset/para_entail_datasets/MCTest/'
    writefile = codecs.open(path+prefix+'_in_entail.txt', 'w', 'utf-8')
    co = 0
    for filename in filenames:
        readfile = codecs.open(path+'Statements/'+filename, 'r', 'utf-8')
        file_content = xmltodict.parse(readfile.read())
        size = len(file_content['devset']['pair'])
        for i in range(size):
            dictt = file_content['devset']['pair'][i]
            # print('dictt:', dictt)
            doc_str = dictt['t']
            sum_str = dictt['h']
            label = dictt['@entailment']
            if label == 'UNKNOWN':
                writefile.write('non_entailment'+'\t'+doc_str.strip()+'\t'+sum_str.strip()+'\n')
            else:
                writefile.write('entailment'+'\t'+doc_str.strip()+'\t'+sum_str.strip()+'\n')
            co+=1
            if co % 50 ==0:
                print('write size:', co)
        readfile.close()
    writefile.close()



        # print(len(doc['devset']['pair']))
        # print(doc['devset']['pair'][0])

def recover_FEVER_dev_test_labels():
    '''
    the nli version of fever does not have label for dev and test; we need to search the gold label
    from the paper version of fever we used before
    but can only retrieval gold label for dev, not test
    '''
    files = ['paper_dev.jsonl', 'paper_test.jsonl']
    id2label = {}
    for fil in files:
        readfile = codecs.open('/export/home/Dataset/FEVER_paper_version/'+fil, 'r', 'utf-8')
        for line in json_lines.reader(readfile):
            id = line.get('id')
            label = line.get('label')
            id2label[str(id)] =  label
        readfile.close()
    readfile = codecs.open('/export/home/Dataset/para_entail_datasets/nli_FEVER/nli_fever/dev_fitems.jsonl', 'r', 'utf-8')
    writefile = codecs.open('/export/home/Dataset/para_entail_datasets/nli_FEVER/nli_fever/dev_fitems.label.recovered.jsonl', 'w', 'utf-8')

    for line in json_lines.reader(readfile):
        id = line.get('cid')
        gold_label = id2label.get(id)
        if gold_label is None:
            print(line)
            exit(0)
        line['label'] = gold_label
        writefile.write(json.dumps(line)+'\n')
    readfile.close()
    writefile.close()
    print('recover over')


def preprocess_curation():
    '''first load summaries'''
    filename = '/export/home/Dataset/Curation_summarization/curation-corpus/curation-corpus-base.csv'
    url2sum = {}
    df = pd.read_csv(filename)
    for i in progress_bar(range(df.shape[0])):
        try:
            url = df.iloc[i][0]
            # print('url:', url)
            # print('headline:', df.iloc[i][1])
            # print('sum:', df.iloc[i][2])
            sum = ' '.join(df.iloc[i][2].strip().split())
        except Exception:
            continue
        url2sum[url] =sum
    print('summary size:', len(url2sum))

    filename = '/export/home/Dataset/Curation_summarization/curation-corpus/curation-corpus-base-with-articles.csv'
    url2doc = {}
    df = pd.read_csv(filename)
    for i in progress_bar(range(df.shape[0])):
        try:
            url_new = df.iloc[i][0]
            soup = BeautifulSoup(Document(df.iloc[i][1]).summary(), features="lxml")

            # delete unwanted tags:
            for e in soup(['figure', 'script']):
                e.decompose()

            doc = ''
            p_list = soup.find_all('p')
            for para in p_list:
                doc+= ' '+para.get_text().strip()
                # print(' '.join(para.get_text().strip().split()))
            full_doc = ' '.join(doc.strip().split())
        except Exception:
            full_doc = "Exception"

        if full_doc != 'Exception':
            url2doc[url_new] = full_doc
    print('doc size:', len(url2doc))

    writefile = codecs.open('/export/home/Dataset/Curation_summarization/curation-corpus/doc_sum.pairs.txt', 'w', 'utf-8')
    valid_size = 0
    for url, doc in url2doc.items():
        sum = url2sum.get(url)
        if sum is not None:
            writefile.write(doc+'\t'+sum+'\n')
            valid_size+=1
    writefile.close()
    print('write  over:', valid_size)


def load_Curation():
    '''
    this function load 40K curation, and gneerate the negative summaries
    '''

    # write_train = codecs.open('/export/home/Dataset/para_entail_datasets/Curation/train_in_entail.txt', 'w', 'utf-8')
    write_dev = codecs.open('/export/home/Dataset/para_entail_datasets/Curation/dev_in_entail.txt', 'w', 'utf-8')
    write_test = codecs.open('/export/home/Dataset/para_entail_datasets/Curation/test_in_entail.txt', 'w', 'utf-8')
    readfile = codecs.open('/export/home/Dataset/Curation_summarization/curation-corpus/doc_sum.pairs.txt', 'r', 'utf-8')# size 39067
    mask_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
    mask_model = AutoModelWithLMHead.from_pretrained("distilbert-base-cased")
    mask_model.to(device)

    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    gpt2_model = AutoModelWithLMHead.from_pretrained("gpt2")
    gpt2_model.to(device)

    size = 0
    prior_unrelated_doc = "Donald John Trump is the 45th and current president of the United States. Before entering politics, he was a businessman and television personality. Trump was born and raised in Queens, a borough of New York City, and received a bachelor's degree in economics from the Wharton School."
    invalid_size = 0
    for line in readfile:
        parts = line.strip().split('\t')
        if len(parts) ==2:
            size+=1
            # print('size:', size)
            # if size < 242:
            #     continue
            if size <=20000:
                continue
                # writefile = write_train
            elif size>20000 and size <=27000:
                # writefile.close()
                writefile = write_dev
            else:
                # writefile.close()
                writefile = write_test

            # print('parts:', parts)
            doc_str = parts[0].strip()
            sum_str = parts[1].strip()
            if len(doc_str.split()) <10 or len(sum_str.split()) < 10:
                invalid_size+=1
                continue

            writefile.write('document>>' +'\t'+doc_str+'\n')
            # writefile.write('positive' +'\t'+doc_str + '\t' + sum_str+'\n')
            writefile.write('positive>>' +'\t'+sum_str+'\n')
            # print('load_DUC_train.prior_unrelated_doc:', prior_unrelated_doc)
            neg_sum_list, neg_sum_namelist = generate_negative_summaries(prior_unrelated_doc, doc_str, sum_str, mask_tokenizer, mask_model, gpt2_tokenizer, gpt2_model)
            prior_unrelated_doc = doc_str
            # print('load_DUC_train.prior_unrelated_doc.update:', prior_unrelated_doc)
            for id, neg_sum in enumerate(neg_sum_list):
                writefile.write('negative>>' +'\t'+neg_sum_namelist[id]+'>>\t'+neg_sum+'\n')
            writefile.write('\n')
            # size+=1
            if size % 10 == 0:
                print('doc size:', size)
        else:
            invalid_size+=1
    writefile.close()
    print('over, invalid_size:', invalid_size)

def split_DUC():
    readfile = codecs.open('/export/home/Dataset/para_entail_datasets/DUC/test_in_entail_original.txt', 'r', 'utf-8')
    writedev = codecs.open('/export/home/Dataset/para_entail_datasets/DUC/dev_in_entail.txt', 'w', 'utf-8')
    writetest = codecs.open('/export/home/Dataset/para_entail_datasets/DUC/test_in_entail.txt', 'w', 'utf-8')
    ex_co = 0
    writefile = writedev
    for line in readfile:
        if len(line.strip()) ==0:
            writefile.write('\n')
            if ex_co < 99:
                writefile = writedev
            else:
                writefile = writetest
            ex_co +=1
        else:
            writefile.write(line.strip()+'\n')

    readfile.close()
    writedev.close()
    writetest.close()



def generate_adversarial_for_summary_data(folder):
    '''
    for our generated negative summary data, we found hypothesis-only bias,
    to remove it, we add some positive hypothesis from other pairs to be negative
    both in training, dev and test. The reason we also do this for training, because we really
    want to the system to learn that a hypothesis might be negative in other premise, but maybe
    positive here
    to test: we create two, one from test, one from training

    '''
    pos_summary_2_doc = {}
    read_train = codecs.open(folder+'/train_in_entail.txt', 'r', 'utf-8')
    # readfile = codecs.open(filename, 'r', 'utf-8')
    start = False
    for line in read_train:
        if len(line.strip()) == 0:
            start = False
        else:
            parts = line.strip().split('\t')
            if parts[0] == 'document>>':
                start = True
                premise = parts[1].strip()
            elif parts[0] == 'positive>>' and start:
                guid_id+=1
                pos_hypo = parts[1].strip()
                if len(premise) == 0 or len(pos_hypo)==0:
                    continue
                else:
                    pos_summary_2_doc[pos_hypo] = premise
            else:
                continue
    read_train.close()

    '''extend training set'''
    read_train = codecs.open(folder+'/train_in_entail.txt', 'r', 'utf-8')
    write_train = codecs.open(folder+'_adversarial_version/train_in_entail.txt', 'w', 'utf-8')

    doc = ''
    for line in read_train:
        if len(line.strip()) !=0:
            write_train.write(line.strip()+'\n')
            parts = line.strip().split('\t')
            if parts[0] == 'document>>':
                doc = parts[1].strip()
        else:
            '''append 5 negative summaries from other pairs'''
            summary_sample_list = random.sample(list(pos_summary_2_doc.keys()), 5)
            for summary_sample in summary_sample_list:
                if pos_summary_2_doc.get(summary_sample) != doc:
                    write_train.write('negative>>' +'\t'+'#FromOtherPair#'+'>>\t'+summary_sample+'\n')
            write_train.write('\n')
            doc = ''
    write_train.close()
    read_train.close()

    '''extend dev and test'''
    filenames = ['dev_in_entail.txt', 'test_in_entail.txt']
    for filename in filenames:
        readfile =










if __name__ == "__main__":
    # mask_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
    # mask_model = AutoModelWithLMHead.from_pretrained("distilbert-base-cased")
    # sum_str = 'to save time, we only use the first summary to generate negative ones'
    # print(random_add_words(sum_str, 0.2, mask_tokenizer, mask_model))

    # load_DUC_train()
    # load_DUC_test()
    # load_CNN_DailyMail()
    load_MCTest(['mc500.train.statements.pairs'], 'mc500.train')
    load_MCTest(['mc500.dev.statements.pairs'], 'mc500.dev')
    load_MCTest(['mc500.test.statements.pairs'], 'mc500.test')

    load_MCTest(['mc160.train.statements.pairs'], 'mc160.train')
    load_MCTest(['mc160.dev.statements.pairs'], 'mc160.dev')
    load_MCTest(['mc160.test.statements.pairs'], 'mc160.test')

    # recover_FEVER_dev_test_labels()

    # preprocess_curation()
    # load_Curation()


    # split_DUC()
    '''
    CUDA_VISIBLE_DEVICES=0
    '''
