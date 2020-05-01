

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
import en_core_web_sm

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

    print('token_list:', token_list)


nlp = en_core_web_sm.load()
summary = 'Donald John Trump is the 45th and current president of the United States.'
sentence_tokenize_by_entities(summary, nlp(summary))
