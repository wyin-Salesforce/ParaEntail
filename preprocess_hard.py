
import codecs
import os
import random
import torch
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
from collections import defaultdict
from transformers import AutoModelWithLMHead, AutoTokenizer
# from transformers import AutoModelWithLMHead, AutoTokenizer
# from transformers import pipeline
import numpy as np
import xmltodict
import json_lines
import json
import csv
import pandas as pd
from fastprogress.fastprogress import progress_bar
# import aiohttp
import asyncio
from bs4 import BeautifulSoup
import csv
from fastprogress.fastprogress import progress_bar
import os
import pandas as pd
from readability import Document
from sys import argv
# from transformers import CTRLLMHeadModel, CTRLTokenizer

seed = 400
random.seed(seed)
np.random.seed(seed)
device = torch.device("cuda")


nlp = en_core_web_sm.load()

def load_CNN_DailyMail(prefix):
    # mask_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
    # mask_model = AutoModelWithLMHead.from_pretrained("distilbert-base-cased")
    mask_tokenizer = AutoTokenizer.from_pretrained("bert-large-cased")
    mask_model = AutoModelWithLMHead.from_pretrained("bert-large-cased")
    mask_model.to(device)

    # gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # gpt2_model = AutoModelWithLMHead.from_pretrained("gpt2")
    # gpt2_model.to(device)
    ctrl_tokenizer = CTRLTokenizer.from_pretrained('ctrl')
    ctrl_model = CTRLLMHeadModel.from_pretrained('ctrl')
    ctrl_model.to(device)

    file_prefix = [prefix]#['train', 'val', 'test']
    for fil_prefix in file_prefix:
        readfil = '/export/home/Dataset/CNN-DailyMail-Summarization/split/'+fil_prefix+'_tokenized.txt'
        writefil = '/export/home/Dataset/para_entail_datasets/CNN_DailyMail/'+fil_prefix+'_in_entail.harsh.v2.txt'
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
                sum_str = parts[1].strip()
                if len(sum_str.split()) > 200:
                    skip_overlong_sum_size+=1
                    continue
                # writefile.write('document>>' +'\t'+doc_str+'\n')
                writefile.write('document>>' +'\t'+'#originalArticle#>>'+'\t'+doc_str+'\n')
                writefile.write('positive>>'+'\t'+'#originalSummaryIsPos#>>' +'\t'+sum_str+'\n')
                neg_sum_list, neg_sum_namelist, neg_sum_list_premise = generate_negative_summaries(prior_unrelated_doc, doc_str, sum_str, mask_tokenizer, mask_model, ctrl_tokenizer, ctrl_model)
                prior_unrelated_doc = doc_str
                for id, neg_sum in enumerate(neg_sum_list):
                    writefile.write('negative>>' +'\t'+neg_sum_namelist[id]+'>>\t'+neg_sum+'\n')
                writefile.write('\n')

                '''
                finish the original doc, now start
                (random_fake --> real) -- negative
                (fake_Plus --> fake) -- positive
                '''
                random_fake_sum = random.choice(neg_sum_list)
                writefile.write('document>>' +'\t'+'#RandomFakeAsPremise#>>'+'\t'+random_fake_sum+'\n')
                writefile.write('negative>>' +'\t'+'#RandomFake2RealIsNeg#'+'>>\t'+sum_str+'\n')
                writefile.write('\n')
                for idd, neg_sum_i in enumerate(neg_sum_list):
                    writefile.write('document>>' +'\t'+'#FakePlusAsPremise#>>'+'\t'+neg_sum_list_premise[idd]+'\n')
                    writefile.write('positive>>'+'\t'+'#FakePlus2FakeIsPos#>>' +'\t'+neg_sum_i+'\n')
                    writefile.write('\n')

                size+=1
                if size % 500 == 0:
                    print(fil_prefix, ' doc size:', size)
        readfile.close()
        writefile.close()
        print('over, size:', size)

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

    preferred_POStags = set(['VERB', 'NOUN', 'PROPN', 'NUM'])
    doc = nlp(sum_str)
    # pos2words = defaultdict(list)
    input_wordlist = []
    input_postag_list = []
    for token in doc:
        # print(token.text, '>>', token.pos_)
        # pos2words[token.pos_].append(token)
        input_wordlist.append(token.text)
        input_postag_list.append(token.pos_)

    # input_wordlist = sum_str.strip().split()
    sum_len = len(input_wordlist)
    replace_size = int(sum_len*drop)#0.3
    replace_size = min(replace_size, 8)

    prior_sum = input_wordlist
    replaced_position_set = set()
    for i in range(replace_size):
        prior_len = len(prior_sum)
        '''start to find a place to replace the word, only if it was replace before, and some pos tags'''
        pos = min(random.randrange(prior_len-1), 200)
        while pos in replaced_position_set or input_postag_list[pos] not in preferred_POStags:
            pos = min(random.randrange(prior_len-1), 200)



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

        replaced_position_set.add(pos)

    return [' '.join(prior_sum)]

def append_unrelated_sents(sum_str, prior_unrelated_doc):
    # nlp = spacy.load('en_core_web_sm')
    # text = "Donald John Trump is the 45th and current president of the United States. Before entering politics, he was a businessman and television personality. Trump was born and raised in Queens, a borough of New York City, and received a bachelor's degree in economics from the Wharton School."
    # text_sentences = nlp(text)
    # for sentence in text_sentences.sents:


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


def insert_unrelated_sents_random_location(sum_str, doc_sents):
    # nlp = spacy.load('en_core_web_sm')
    # text = "Donald John Trump is the 45th and current president of the United States. Before entering politics, he was a businessman and television personality. Trump was born and raised in Queens, a borough of New York City, and received a bachelor's degree in economics from the Wharton School."
    # text_sentences = nlp(text)
    # for sentence in text_sentences.sents:


    text_sentences = nlp(sum_str)
    sum_sents = []
    for sentence in text_sentences.sents:
        sum_sents.append(sentence.text)

    random_sent_from_doc = random.choice(doc_sents)
    '''put the unrelated sent at the position 1'''
    random_position = random.sample(list(range(len(sum_sents))), 1)[0]
    new_sum_sents = sum_sents[:random_position]+[random_sent_from_doc]+sum_sents[random_position:]

    fake_summary_plus = ' '.join(new_sum_sents)

    return ' '.join(fake_summary_plus.strip().split())

def CTRL_generate(sum_str, tokenizer, model, replace = False):

    text_sentences = nlp(sum_str)
    sum_sents = []
    for sentence in text_sentences.sents:
        sum_sents.append(sentence.text) # string

    sent_size = len(sum_sents)
    max_len = len(sum_str.split())+20

    # keep_lengths = [int(input_len*0.3), int(input_len*0.6), int(input_len*0.9)]
    new_seqs = []
    if sent_size <2:
        return new_seqs
    for _ in range(1):
        '''which sentence to split'''
        sent_id = random.sample(list(range(1, sent_size)), 1)[0]
        # sent_len = len(sum_sents[sent_id].split())
        '''which word to split'''
        # word_id = random.sample(list(range(sent_len)), 1)[0]
        know_word_list = []
        for i in range(sent_id):
            for word in sum_sents[i].split():
                know_word_list.append(word)
        kept_sent = sum_sents[:sent_id]
        if replace:
            remaining_sents = sum_sents[sent_id+1:]
        else:
            '''insert, we keep all the original sentences'''
            remaining_sents = sum_sents[sent_id:]


        prompt_text = 'Links '+' '.join(kept_sent)# if args.prompt else input("Model prompt >>> ")
        # preprocessed_prompt_text = prompt_text#prepare_ctrl_input(args, model, tokenizer, prompt_text)
        encoded_prompt = tokenizer.encode(
            prompt_text, add_special_tokens=False, return_tensors="pt", add_space_before_punct_symbol=True
            )
        encoded_prompt = encoded_prompt.to(device)
        output_sequences = model.generate(
            input_ids=encoded_prompt,
            max_length=40 + len(encoded_prompt[0]),
            temperature=0.1,
            top_k=0,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True,
            num_return_sequences=1,
        )
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()
        generated_sequences = []
        for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
            # print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
            generated_sequence = generated_sequence.tolist()
            text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
            # text = text[: text.find(args.stop_token) if args.stop_token else None]
            # text = text[: text.find(None) if None else None]
            # total_sequence = (
            #     prompt_text + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
            #     )

            generate_part = text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]

            # print('prompt_text:', prompt_text)
            # print('generate_part:', generate_part)
            # # exit(0)
            generated_sequences.append(generate_part.strip())
            break

        resulting_sentences = nlp(generated_sequences[0])
        selected_sent = []
        for new_sentence in resulting_sentences.sents:
            selected_sent.append(new_sentence.text) # string
            break

        # print('sent_id:', sent_id)
        # print('resulting_sents:', resulting_sents)
        # selected_sent = resulting_sents#[resulting_sents[0]]

        new_seq = ' '.join(kept_sent+selected_sent+remaining_sents)
        new_seq = ' '.join(new_seq.strip().split())

        new_seqs.append(new_seq)
    # print(new_seqs)

    return new_seqs



# def GPT2_generate(sum_str, tokenizer, model):
#     text_sentences = nlp(sum_str)
#     sum_sents = []
#     for sentence in text_sentences.sents:
#         sum_sents.append(sentence.text) # string
#
#     sent_size = len(sum_sents)
#
#
#
#     print('sum_sents:', sum_sents)
#
#     # input_wordlist = sum_str.split()
#     # input_len = len(input_wordlist)
#     max_len = len(sum_str.split())+20
#
#     # keep_lengths = [int(input_len*0.3), int(input_len*0.6), int(input_len*0.9)]
#     new_seqs = []
#     if sent_size <2:
#         return new_seqs
#     for _ in range(1):
#         '''which sentence to split'''
#         sent_id = random.sample(list(range(1, sent_size)), 1)[0]
#         # sent_len = len(sum_sents[sent_id].split())
#         '''which word to split'''
#         # word_id = random.sample(list(range(sent_len)), 1)[0]
#         know_word_list = []
#         for i in range(sent_id):
#             for word in sum_sents[i].split():
#                 know_word_list.append(word)
#         kept_sent = sum_sents[:sent_id]
#         remaining_sents = sum_sents[sent_id+1:]
#
#         # print('know_word_list:', know_word_list)
#
#         sequence = ' '.join(know_word_list)#f"Hugging Face is based in DUMBO, New York City, and is"
#         # print('sequence:', sequence)
#         input = tokenizer.encode(sequence, return_tensors="pt")
#         input = input.to(device)
#         # print('input:', input)
#         generated = model.generate(input, max_length=max_len)
#
#         resulting_string = tokenizer.decode(generated.tolist()[0]).strip()
#
#         resulting_sentences = nlp(resulting_string)
#         resulting_sents = []
#         for new_sentence in resulting_sentences.sents:
#             resulting_sents.append(new_sentence.text) # string
#
#         print('sent_id:', sent_id)
#         print('resulting_sents:', resulting_sents)
#         selected_sent = [resulting_sents[sent_id]]
#
#         new_seq = kept_sent+selected_sent+remaining_sents
#         # # new_seq = know_word_list
#         # print('new_seq:', new_seq)
#         # exit(0)
#
#         new_seqs.append(' '.join(new_seq))
#     # print(new_seqs)
#
#     return new_seqs

def sentence_tokenize_by_entities(sentence, nlp_proprocess):
    '''
    #nlp_proprocess = nlp(sentence)
    '''
    token_list = []

    last_end = 0
    for ent in nlp_proprocess.ents:
        # print(ent.text, ent.start_char, ent.end_char, ent.label_)
        left_context = sentence[last_end:ent.start_char]
        token_list+= left_context.strip().split()
        token_list+=[ent.text]
        last_end = ent.end_char
    token_list+= sentence[last_end:].strip().split()
    return token_list



def replace_N_entities_by_NER(article, summary_str):

    summary = nlp(summary_str)
    summary_entityToken_list = sentence_tokenize_by_entities(summary_str, summary)
    nerlabel2entitylist = {}
    entity_2_label = {}
    for X in summary.ents:
        entlist = nerlabel2entitylist.get(X.label_)
        if entlist is None:
            entlist = []
        entlist.append(X.text)
        entity_2_label[X.text] = X.label_
        nerlabel2entitylist[X.label_] = entlist

    if len(entity_2_label.keys()) < 5:
        return False

    random_N_entities = random.sample(list(entity_2_label.keys()), 5)

    nerlabel2entitylist_doc = {}
    for ent in random_N_entities:
        if len(nerlabel2entitylist.get(entity_2_label.get(ent))) == 1:
            doc = nlp(article)
            for X in doc.ents:
                entlist = nerlabel2entitylist_doc.get(X.label_)
                if entlist is None:
                    entlist = []
                entlist.append(X.text)
                nerlabel2entitylist_doc[X.label_] = entlist
            break


    replaced_indices = set()
    for ent in random_N_entities:
        # ind = summary_entityToken_list.index(ent)
        indices = [i for i, x in enumerate(summary_entityToken_list) if x == ent]
        ind = random.choice(indices)
        while ind in replaced_indices:
            ind = random.choice(indices)

        ent_label = entity_2_label.get(ent)
        entities_in_the_same_group = set(nerlabel2entitylist.get(ent_label)) - set([ent])
        if len(entities_in_the_same_group) > 0:
            new_ent = random.choice(list(entities_in_the_same_group))
        else:
            # print('ent_label:', ent_label)
            # print('nerlabel2entitylist:', nerlabel2entitylist)
            # print('nerlabel2entitylist_doc:', nerlabel2entitylist_doc)
            # print(set(nerlabel2entitylist_doc.get(ent_label)))
            # print(set([ent]))
            all_similar_entities_from_article = nerlabel2entitylist_doc.get(ent_label)
            if all_similar_entities_from_article is None:
                continue
            entities_from_article = set(all_similar_entities_from_article) - set([ent])
            if len(entities_from_article) < 1:
                continue
            else:
                new_ent = random.choice(list(entities_from_article))
        summary_entityToken_list[ind] = new_ent
        replaced_indices.add(ind)

    new_summary = ' '.join(summary_entityToken_list)

    # print('old summary:', summary_str)
    # print('new_summary:', new_summary)
    # exit(0)
    return new_summary




def NER(input):


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
    # entity_cand_list = swap_entities(doc_str, sum_str)
    entity_cand_list = []
    entity_cand_list_names = []
    fake_entity_swapped_summary = replace_N_entities_by_NER(doc_str, sum_str)
    if fake_entity_swapped_summary is not False:
        entity_cand_list = [fake_entity_swapped_summary]
        entity_cand_list_names = ['#EntityReplacedIsNeg#']
    '''word-level noise'''
    bert_mask_list = random_replace_words(sum_str, 0.2, mask_tokenizer, mask_model)
    bert_mask_list_names = ['#WordReplacedIsNeg#'] * len(bert_mask_list)

    '''sentence-level noise'''
    bert_generate_list = CTRL_generate(sum_str, gpt2_tokenizer, gpt2_model, replace=True)
    bert_generate_list_names = ['#SentReplacedIsNeg#'] * len(bert_generate_list)

    cand_list= entity_cand_list  + bert_mask_list +bert_generate_list
    name_list = entity_cand_list_names + bert_mask_list_names + bert_generate_list_names

    '''now, for all fake summaries, we insert a sentence selected from the article'''
    premise_cand_list = []

    doc_sentences = nlp(doc_str)
    doc_sents = []
    for sentence in doc_sentences.sents:
        doc_sents.append(sentence.text)
    for cand_i in cand_list:
        # print('haha...')
        cand_i_premise = insert_unrelated_sents_random_location(cand_i, doc_sents)
        # print('haha')
        # cand_i_premise = CTRL_generate(cand_i, gpt2_tokenizer, gpt2_model, replace=False)
        premise_cand_list.append(cand_i_premise)


    return cand_list, name_list, premise_cand_list

def load_DUC_train():
    #DUC2001
    trainfolder_namelist = ['d01a','d02a','d03a','d07b','d09b','d10b','d16c','d17c','d18c','d20d','d21d',
    'd23d','d25e','d26e','d29e','d33f','d35f','d36f','d38g','d40g','d42g','d46h','d47h','d48h','d49i',
    'd51i','d52i','d55k','d58k','d60k']

    writefile = codecs.open('/export/home/Dataset/para_entail_datasets/DUC/train_in_entail.harsh.v2.txt', 'w', 'utf-8')
    # mask_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
    # mask_model = AutoModelWithLMHead.from_pretrained("distilbert-base-cased")
    mask_tokenizer = AutoTokenizer.from_pretrained("bert-large-cased")
    mask_model = AutoModelWithLMHead.from_pretrained("bert-large-cased")
    mask_model.to(device)

    # gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # gpt2_model = AutoModelWithLMHead.from_pretrained("gpt2")
    # gpt2_model.to(device)
    ctrl_tokenizer = CTRLTokenizer.from_pretrained('ctrl')
    ctrl_model = CTRLLMHeadModel.from_pretrained('ctrl')
    ctrl_model.to(device)

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
        # print('start scan all docs....')
        for id, doc in id2doc.items():
            # print(id, '\n', doc, '\n', id2sum.get(id))
            doc_str = doc

            summ = id2sum.get(id)
            if summ is None or len(doc_str.strip()) == 0:
                print('missing:', foldername, id)
                continue

            writefile.write('document>>' +'\t'+'#originalArticle#>>'+'\t'+doc_str+'\n')
            sum_str = ' '.join(summ.strip().split())

            writefile.write('positive>>'+'\t'+'#originalSummaryIsPos#>>' +'\t'+sum_str+'\n')
            # print('load_DUC_train.prior_unrelated_doc:', prior_unrelated_doc)
            neg_sum_list, neg_sum_namelist, neg_sum_list_premise = generate_negative_summaries(prior_unrelated_doc, doc_str, sum_str, mask_tokenizer, mask_model, ctrl_tokenizer, ctrl_model)
            prior_unrelated_doc = doc_str
            for id, neg_sum in enumerate(neg_sum_list):
                writefile.write('negative>>' +'\t'+neg_sum_namelist[id]+'>>\t'+neg_sum+'\n')
            writefile.write('\n')

            '''
            finish the original doc, now start
            (random_fake --> real) -- negative
            (fake_Plus --> fake) -- positive
            '''
            random_fake_sum = random.choice(neg_sum_list)
            writefile.write('document>>' +'\t'+'#RandomFakeAsPremise#>>'+'\t'+random_fake_sum+'\n')
            writefile.write('negative>>' +'\t'+'#RandomFake2RealIsNeg#'+'>>\t'+sum_str+'\n')
            writefile.write('\n')
            for idd, neg_sum_i in enumerate(neg_sum_list):
                writefile.write('document>>' +'\t'+'#FakePlusAsPremise#>>'+'\t'+neg_sum_list_premise[idd]+'\n')
                writefile.write('positive>>'+'\t'+'#FakePlus2FakeIsPos#>>' +'\t'+neg_sum_i+'\n')
                writefile.write('\n')

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

    writefile = codecs.open('/export/home/Dataset/para_entail_datasets/DUC/test_in_entail.harsh.v2.txt', 'w', 'utf-8')
    # mask_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
    # mask_model = AutoModelWithLMHead.from_pretrained("distilbert-base-cased")
    mask_tokenizer = AutoTokenizer.from_pretrained("bert-large-cased")
    mask_model = AutoModelWithLMHead.from_pretrained("bert-large-cased")
    mask_model.to(device)

    # gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # gpt2_model = AutoModelWithLMHead.from_pretrained("gpt2")
    # gpt2_model.to(device)
    ctrl_tokenizer = CTRLTokenizer.from_pretrained('ctrl')
    ctrl_model = CTRLLMHeadModel.from_pretrained('ctrl')
    ctrl_model.to(device)

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

            # writefile.write('document>>' +'\t'+doc_str+'\n')
            writefile.write('document>>' +'\t'+'#originalArticle#>>'+'\t'+doc_str+'\n')
            for summm in summ_list:
                sum_str = ' '.join(summm.strip().split())
                # writefile.write('positive' +'\t'+doc_str + '\t' + sum_str+'\n')
                # writefile.write('positive>>' +'\t'+sum_str+'\n')
                writefile.write('positive>>'+'\t'+'#originalSummaryIsPos#>>' +'\t'+sum_str+'\n')

            '''to save time, we only use the first summary to generate negative ones'''
            sum_str = ' '.join(summ_list[0].strip().split())
            neg_sum_list, neg_sum_namelist, neg_sum_list_premise = generate_negative_summaries(prior_unrelated_doc, doc_str, sum_str, mask_tokenizer, mask_model, ctrl_tokenizer, ctrl_model)
            prior_unrelated_doc = doc_str
            for id, neg_sum in enumerate(neg_sum_list):
                writefile.write('negative>>' +'\t'+neg_sum_namelist[id]+'>>\t'+neg_sum+'\n')
            writefile.write('\n')

            '''
            finish the original doc, now start
            (random_fake --> real) -- negative
            (fake_Plus --> fake) -- positive
            '''
            random_fake_sum = random.choice(neg_sum_list)
            writefile.write('document>>' +'\t'+'#RandomFakeAsPremise#>>'+'\t'+random_fake_sum+'\n')
            writefile.write('negative>>' +'\t'+'#RandomFake2RealIsNeg#'+'>>\t'+sum_str+'\n')
            writefile.write('\n')
            for idd, neg_sum_i in enumerate(neg_sum_list):
                writefile.write('document>>' +'\t'+'#FakePlusAsPremise#>>'+'\t'+neg_sum_list_premise[idd]+'\n')
                writefile.write('positive>>'+'\t'+'#FakePlus2FakeIsPos#>>' +'\t'+neg_sum_i+'\n')
                writefile.write('\n')

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


def load_Curation(prefix):
    '''
    this function load 40K curation, and gneerate the negative summaries
    '''

    writefile = codecs.open('/export/home/Dataset/para_entail_datasets/Curation/'+prefix+'_in_entail.harsh.v2.txt', 'w', 'utf-8')
    # write_dev = codecs.open('/export/home/Dataset/para_entail_datasets/Curation/dev_in_entail.harsh.txt', 'w', 'utf-8')
    # write_test = codecs.open('/export/home/Dataset/para_entail_datasets/Curation/test_in_entail.harsh.txt', 'w', 'utf-8')
    readfile = codecs.open('/export/home/Dataset/Curation_summarization/curation-corpus/doc_sum.pairs.txt', 'r', 'utf-8')# size 39067
    # mask_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
    # mask_model = AutoModelWithLMHead.from_pretrained("distilbert-base-cased")
    mask_tokenizer = AutoTokenizer.from_pretrained("bert-large-cased")
    mask_model = AutoModelWithLMHead.from_pretrained("bert-large-cased")
    mask_model.to(device)

    # gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # gpt2_model = AutoModelWithLMHead.from_pretrained("gpt2")
    # gpt2_model.to(device)

    ctrl_tokenizer = CTRLTokenizer.from_pretrained('ctrl')
    ctrl_model = CTRLLMHeadModel.from_pretrained('ctrl')
    ctrl_model.to(device)

    size = 0
    prior_unrelated_doc = "Donald John Trump is the 45th and current president of the United States. Before entering politics, he was a businessman and television personality. Trump was born and raised in Queens, a borough of New York City, and received a bachelor's degree in economics from the Wharton School."
    invalid_size = 0
    for line in readfile:
        parts = line.strip().split('\t')
        if len(parts) ==2:
            # print('parts:', parts)
            doc_str = parts[0].strip()
            sum_str = parts[1].strip()
            if len(doc_str.split()) <10 or len(sum_str.split()) < 10:
                invalid_size+=1
                continue
            else:
                '''valid'''
                size+=1
                if prefix == 'train':
                    if size > 20000:
                        break
                elif prefix == 'dev':
                    if size <=20000:
                        continue
                    elif size > 27000:
                        break
                else:
                    if size <=27000:
                        continue


                writefile.write('document>>' +'\t'+'#originalArticle#>>'+'\t'+doc_str+'\n')
                # writefile.write('positive' +'\t'+doc_str + '\t' + sum_str+'\n')
                # writefile.write('positive>>' +'\t'+sum_str+'\n')
                writefile.write('positive>>'+'\t'+'#originalSummaryIsPos#>>' +'\t'+sum_str+'\n')
                # print('load_DUC_train.prior_unrelated_doc:', prior_unrelated_doc)
                neg_sum_list, neg_sum_namelist, neg_sum_list_premise = generate_negative_summaries(prior_unrelated_doc, doc_str, sum_str, mask_tokenizer, mask_model, ctrl_tokenizer, ctrl_model)
                prior_unrelated_doc = doc_str
                for id, neg_sum in enumerate(neg_sum_list):
                    writefile.write('negative>>' +'\t'+neg_sum_namelist[id]+'>>\t'+neg_sum+'\n')
                writefile.write('\n')

                '''
                finish the original doc, now start
                (random_fake --> real) -- negative
                (fake_Plus --> fake) -- positive
                '''
                random_fake_sum = random.choice(neg_sum_list)
                writefile.write('document>>' +'\t'+'#RandomFakeAsPremise#>>'+'\t'+random_fake_sum+'\n')
                writefile.write('negative>>' +'\t'+'#RandomFake2RealIsNeg#'+'>>\t'+sum_str+'\n')
                writefile.write('\n')
                for idd, neg_sum_i in enumerate(neg_sum_list):
                    writefile.write('document>>' +'\t'+'#FakePlusAsPremise#>>'+'\t'+neg_sum_list_premise[idd]+'\n')
                    writefile.write('positive>>'+'\t'+'#FakePlus2FakeIsPos#>>' +'\t'+neg_sum_i+'\n')
                    writefile.write('\n')

                if size % 10 == 0:
                    print('doc size:', size)
        else:
            invalid_size+=1
    writefile.close()
    print('over, invalid_size:', invalid_size)


def preprocess_SQUAD_NLI():
    path = '/export/home/Dataset/SQUAD_2_NLI/'
    question_id2doc = {}
    question_id2answerable = {}
    files = ['train-v2.0.json', 'dev-v2.0.json']
    for fil in files:
        readfile = codecs.open(path+fil, 'r', 'utf-8')
        data = json.load(readfile)
        for p in data['data']:
            for paragraph in p['paragraphs']: # list
                doc = paragraph['context']
                for qas in paragraph['qas']: # list
                    question = qas['question']
                    idd = qas['id']
                    unswerable = qas['is_impossible']
                    print('unswerable:', unswerable)
                    assert unswerable == False
                    question_id2doc[idd] = doc
                    question_id2answerable[idd] = True if unswerable=='False' else False

                    # print('question:', question)
                    # print('idd:', idd)
                    # print('unswerable:', unswerable)

                print('doc:', doc)

                exit(0)

# def load_Curation():
#     '''
#     this function load 40K curation, and gneerate the negative summaries
#     '''
#
#     write_train = codecs.open('/export/home/Dataset/para_entail_datasets/Curation/train_in_entail.harsh.txt', 'w', 'utf-8')
#     write_dev = codecs.open('/export/home/Dataset/para_entail_datasets/Curation/dev_in_entail.harsh.txt', 'w', 'utf-8')
#     write_test = codecs.open('/export/home/Dataset/para_entail_datasets/Curation/test_in_entail.harsh.txt', 'w', 'utf-8')
#     readfile = codecs.open('/export/home/Dataset/Curation_summarization/curation-corpus/doc_sum.pairs.txt', 'r', 'utf-8')# size 39067
#     mask_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
#     mask_model = AutoModelWithLMHead.from_pretrained("distilbert-base-cased")
#     mask_model.to(device)
#
#     # gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
#     # gpt2_model = AutoModelWithLMHead.from_pretrained("gpt2")
#     # gpt2_model.to(device)
#
#     ctrl_tokenizer = CTRLTokenizer.from_pretrained('ctrl')
#     ctrl_model = CTRLLMHeadModel.from_pretrained('ctrl')
#     ctrl_model.to(device)
#
#     size = 0
#     prior_unrelated_doc = "Donald John Trump is the 45th and current president of the United States. Before entering politics, he was a businessman and television personality. Trump was born and raised in Queens, a borough of New York City, and received a bachelor's degree in economics from the Wharton School."
#     invalid_size = 0
#     for line in readfile:
#         parts = line.strip().split('\t')
#         if len(parts) ==2:
#             size+=1
#             # print('size:', size)
#             # if size < 242:
#             #     continue
#             if size <=20000:
#                 continue
#                 # writefile = write_train
#             elif size>20000 and size <=27000:
#                 # writefile.close()
#                 writefile = write_dev
#             else:
#                 # writefile.close()
#                 writefile = write_test
#
#             # print('parts:', parts)
#             doc_str = parts[0].strip()
#             sum_str = parts[1].strip()
#             if len(doc_str.split()) <10 or len(sum_str.split()) < 10:
#                 invalid_size+=1
#                 continue
#
#             writefile.write('document>>' +'\t'+doc_str+'\n')
#             # writefile.write('positive' +'\t'+doc_str + '\t' + sum_str+'\n')
#             writefile.write('positive>>' +'\t'+sum_str+'\n')
#             # print('load_DUC_train.prior_unrelated_doc:', prior_unrelated_doc)
#             neg_sum_list, neg_sum_namelist = generate_negative_summaries(prior_unrelated_doc, doc_str, sum_str, mask_tokenizer, mask_model, ctrl_tokenizer, ctrl_model)
#             prior_unrelated_doc = doc_str
#             # print('load_DUC_train.prior_unrelated_doc.update:', prior_unrelated_doc)
#             for id, neg_sum in enumerate(neg_sum_list):
#                 writefile.write('negative>>' +'\t'+neg_sum_namelist[id]+'>>\t'+neg_sum+'\n')
#             writefile.write('\n')
#             # size+=1
#             if size % 10 == 0:
#                 print('doc size:', size)
#         else:
#             invalid_size+=1
#     writefile.close()
#     print('over, invalid_size:', invalid_size)

def split_DUC():
    readfile = codecs.open('/export/home/Dataset/para_entail_datasets/DUC/test_in_entail.harsh.original.v2.txt', 'r', 'utf-8')
    writedev = codecs.open('/export/home/Dataset/para_entail_datasets/DUC/dev_in_entail.harsh.v2.txt', 'w', 'utf-8')
    writetest = codecs.open('/export/home/Dataset/para_entail_datasets/DUC/test_in_entail.harsh.v2.txt', 'w', 'utf-8')
    ex_co = 0
    writefile = writedev
    for line in readfile:
        if len(line.strip()) ==0:
            rand_prob = random.uniform(0, 1)
            if rand_prob < 1/3:
                writefile = writedev
            else:
                writefile = writetest

            writefile.write('\n')
            ex_co +=1
        else:
            writefile.write(line.strip()+'\n')

    readfile.close()
    writedev.close()
    writetest.close()


def flaging_a_block(block_line_list):
    '''
    return 1 if this block has '#originalSummaryIsPos#>>'
    return a neg_size to denote how many fake summaries
    '''
    # print('block_line_list', block_line_list)
    if len(block_line_list) == 1:
        return 0,0
    second_line_parts = block_line_list[1].strip().split('\t')
    if second_line_parts[1] == '#originalSummaryIsPos#>>':
        fake_size = 0
        for line in block_line_list[2:]:
            line_parts = line.strip().split('\t')
            if line_parts[1] in set(['#SwapEnt#>>', '#ReplaceWord#>>', '#ReplaceUnrelatedSent#>>']):
                fake_size+=1

        return 1, fake_size*2
    else:
        return 0, 0

# def get_ensembled_summary_by_entity_swap(swap_entity_summary_list):




if __name__ == "__main__":

    # load_DUC_train()
    # load_DUC_test()

    # load_CNN_DailyMail('train')
    # load_CNN_DailyMail('val')
    # load_CNN_DailyMail('test')

    # load_MCTest(['mc500.train.statements.pairs', 'mc160.train.statements.pairs'], 'train')
    # load_MCTest(['mc500.dev.statements.pairs', 'mc160.dev.statements.pairs'], 'dev')
    # load_MCTest(['mc500.test.statements.pairs', 'mc160.test.statements.pairs'], 'test')

    # recover_FEVER_dev_test_labels()

    # preprocess_curation()
    # load_Curation('train')
    # load_Curation('dev')
    # load_Curation('test')


    # split_DUC()

    '''preprocess QA into NLI'''
    preprocess_SQUAD_NLI()




    '''
    CUDA_VISIBLE_DEVICES=0
    '''
