
import codecs
import os
import nltk
from nltk import pos_tag, ne_chunk
from nltk.tokenize import word_tokenize
from nltk.chunk import conlltags2tree, tree2conlltags
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
    readfile.close()
    return doc.strip()

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

def shuffle_words_same_POStags(sum_str):
    nlp = en_core_web_sm.load()
    doc = nlp(sum_str)
    pos2words = defaultdict(list)
    for token in doc:
        pos2words[token.pos_].append(token)
    new_word_list = []
    for token in doc:
        word_set = set(pos2words.get(token.pos_))
        if len(word_set) ==  1:
            new_word_list.append(token.text)
            continue
        else:
            word_set.discard(token)
            assert len(word_set) >=1
            prob = random.uniform(0, 1)
            if prob < 0.3:
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

    new_tokens = []
    for word in tokens:
        prob = random.uniform(0, 1)
        if prob < drop:
            #0.8
            new_tokens.append(word)
        else:
            continue

    return [' '.join(new_tokens)]

def random_add_words(sum_str, drop, tokenizer, model):

    # nlp = pipeline('fill-mask', 'checkpoint-335000/', tokenizer='roberta-large')
    # print(nlp('On admission, the most common symptoms were <mask>'))
    #
    # from transformers import pipeline
    # print(sum_str)

    input_wordlist = sum_str.strip().split()
    sum_len = len(input_wordlist)
    insert_size = int(sum_len*drop)#0.3

    prior_sum = input_wordlist
    for i in range(insert_size):
        prior_len = len(prior_sum)
        pos = random.randrange(prior_len-1)
        sequence = ' '.join(prior_sum[:pos])+' '+ f"{tokenizer.mask_token}" + ' '+ ' '.join(prior_sum[pos:])




        # tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
        # model = AutoModelWithLMHead.from_pretrained("distilbert-base-cased")

        # sequence = f"Distilled models are smaller than the models they mimic. Using them instead of the large versions would help {tokenizer.mask_token} our carbon footprint."

        input = tokenizer.encode(sequence, return_tensors="pt")
        mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]

        token_logits = model(input)[0]
        mask_token_logits = token_logits[0, mask_token_index, :]

        top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
        # print(top_5_tokens)

        prior_sum = sequence.replace(tokenizer.mask_token, tokenizer.decode([top_5_tokens[0]])).split()
        # print(' '.join(prior_sum))

    return [' '.join(prior_sum)]

def append_unrelated_sents(sum_str, source_sent_list):
    return

def GPT2_generate(sum_str, tokenizer, model):
    input_wordlist = sum_str.split()
    input_len = len(input_wordlist)
    max_len = input_len+15

    keep_lengths = [int(input_len*0.3), int(input_len*0.6), int(input_len*0.9)]
    new_seqs = []
    for leng in keep_lengths:

        sequence = ' '.join(input_wordlist[:leng])#f"Hugging Face is based in DUMBO, New York City, and is"

        input = tokenizer.encode(sequence, return_tensors="pt")
        generated = model.generate(input, max_length=max_len)

        resulting_string = tokenizer.decode(generated.tolist()[0])
        # print('resulting_string:', resulting_string)
        new_seq = resulting_string[:resulting_string.rfind('.')+1]
        # print(resulting_string.rfind('.'), len(sum_str))
        if resulting_string.rfind('.') < len(sum_str):
            continue
        else:
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

def generate_negative_summaries(doc_str, sum_str, mask_tokenizer, mask_model, gpt2_tokenizer, gpt2_model):
    entity_cand_list = swap_entities(doc_str, sum_str)
    # swap_pronouns(doc_str, sum_str)
    shuffle_word_list = shuffle_words_same_POStags(sum_str)
    missing_word_list = random_remove_words(sum_str, 0.8)
    bert_mask_list = random_add_words(sum_str, 0.3, mask_tokenizer, mask_model)
    # append_unrelated_sents(sum_str, source_sent_list)
    bert_generate_list = GPT2_generate(sum_str, gpt2_tokenizer, gpt2_model)
    return entity_cand_list + shuffle_word_list + missing_word_list + bert_mask_list + bert_generate_list


def load_DUC_train():
    #DUC2001
    trainfolder_namelist = ['d01a','d02a','d03a','d07b','d09b','d10b','d16c','d17c','d18c','d20d','d21d',
    'd23d','d25e','d26e','d29e','d33f','d35f','d36f','d38g','d40g','d42g','d46h','d47h','d48h','d49i',
    'd51i','d52i','d55k','d58k','d60k']

    writefile = codecs.open('/export/home/Dataset/para_entail_datasets/DUC/train_in_entail.txt', 'w', 'utf-8')
    mask_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
    mask_model = AutoModelWithLMHead.from_pretrained("distilbert-base-cased")

    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    gpt2_model = AutoModelWithLMHead.from_pretrained("gpt2")

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

        for id, doc in id2doc.items():
            # print(id, '\n', doc, '\n', id2sum.get(id))
            doc_str = ' '.join(doc.strip().split())
            summ = id2sum.get(id)
            if summ is None:
                print('missing:', foldername, id)
                continue
            sum_str = ' '.join(summ.strip().split())

            writefile.write('positive' +'\t'+doc_str + '\t' + sum_str+'\n')
            neg_sum_list = generate_negative_summaries(doc_str, sum_str, mask_tokenizer, mask_model, gpt2_tokenizer, gpt2_model)
            for neg_sum in neg_sum_list:
                writefile.write('negative' +'\t'+doc_str + '\t' + neg_sum+'\n')
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

    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    gpt2_model = AutoModelWithLMHead.from_pretrained("gpt2")

    '''test doc has multiple summary'''
    duplicate_sum_pathstring = '/export/home/Dataset/para_entail_datasets/DUC/DUC_data/data/duc01/data/test/duplicate.summaries'
    duplicate_sum_path = os.listdir(duplicate_sum_pathstring)
    folder_2_multiple = defaultdict(list)
    for foldername in duplicate_sum_path:
        path_to_folder = os.path.join(duplicate_sum_pathstring, foldername)
        if os.path.isdir(path_to_folder):
            folder_2_multiple[foldername[:4]].append(foldername)

    print('folder_2_multiple:', folder_2_multiple)
    exit(0)
    size = 0
    for foldername in test_folder_namelist:
        last_char = foldername[-1]
        subfolder = foldername+last_char
        docsfolder = 'docs'

        # /export/home/Dataset/para_entail_datasets/DUC/DUC_data/data/duc01/data/test/original.summaries/d56kk
        perdoc_file ='/export/home/Dataset/para_entail_datasets/DUC/DUC_data/data/duc01/data/test/original.summaries'+'/'+subfolder+'/perdocs'
        id2sum = load_per_docs_file(perdoc_file)
        '''load duplicate summary'''
        id2sumlist = defaultdict(list)
        for idd, sum_i in id2sum.items():
            id2sumlist[idd].append(sum_i)
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

        for id, doc in id2doc.items():
            # print(id, '\n', doc, '\n', id2sum.get(id))
            doc_str = ' '.join(doc.strip().split())
            summ_list = id2sumlist.get(id)
            if summ_list is None:
                print('missing:', foldername, id)
                continue

            for summm in summ_list:
                sum_str = ' '.join(summm.strip().split())
                writefile.write('positive' +'\t'+doc_str + '\t' + sum_str+'\n')

            '''to save time, we only use the first summary to generate negative ones'''
            sum_str = ' '.join(summ_list[0].strip().split())
            neg_sum_list = generate_negative_summaries(doc_str, sum_str, mask_tokenizer, mask_model, gpt2_tokenizer, gpt2_model)
            for neg_sum in neg_sum_list:
                writefile.write('negative' +'\t'+doc_str + '\t' + neg_sum+'\n')
            size+=1
            if size % 10 == 0:
                print('doc size:', size)
    writefile.close()





if __name__ == "__main__":
    # load_per_docs_file('/export/home/Dataset/para_entail_datasets/DUC/DUC_data/data/duc01/data/training/d49i/d49ii/perdocs')
    # load_DUC_train()
    load_DUC_test()
    # NER('European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices.')
    # appearance_of_str('why we do there without why you come why why .', 'why')
    # shuffle_words_same_POStags('Salesforce is located in San Francisco, California, why you join it')
    # tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
    # model = AutoModelWithLMHead.from_pretrained("distilbert-base-cased")
    # random_add_words('Distilled models are smaller than the models they mimic. Using them instead of the large versions would help our carbon footprint.', 0.3, tokenizer, model)

    # tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # model = AutoModelWithLMHead.from_pretrained("gpt2")
    # GPT2_generate('Distilled models are smaller than the models they mimic. Using them instead of the large versions would help our carbon footprint.', tokenizer, model)


    '''
    CUDA_VISIBLE_DEVICES=0
    '''
