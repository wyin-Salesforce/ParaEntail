
def get_DUC_examples(filename):
    #/export/home/Dataset/para_entail_datasets/DUC/train_in_entail.txt
    print('loading...', filename)
    readfile = codecs.open(filename, 'r', 'utf-8')
    start = False
    examples = []
    guid_id = -1
    pos_size = 0
    neg_size = 0
    for line in readfile:
        if len(line.strip()) == 0:
            start = False
        else:
            parts = line.strip().split('\t')
            if parts[0] == 'document>>':
                start = True
                premise = parts[1].strip()
            elif parts[0] == 'positive>>':
                guid_id+=1
                pos_hypo = parts[1].strip()
                examples.append(InputExample(guid=str(guid_id), text_a=premise, text_b=pos_hypo, label='entailment'))
                pos_size+=1
            elif parts[0] == 'negative>>' and parts[1] != '#ShuffleWord#>>' and parts[1] != '#RemoveWord#>>':
                guid_id+=1
                neg_hypo = parts[2].strip()

                if filename.find('train_in_entail') > -1:
                    examples.append(InputExample(guid=str(guid_id), text_a=premise, text_b=neg_hypo, label='not_entailment'))
                    neg_size+=1
                else:
                    rand_prob = random.uniform(0, 1)
                    if rand_prob > 3/4:
                        examples.append(InputExample(guid=str(guid_id), text_a=premise, text_b=neg_hypo, label='not_entailment'))
                        neg_size+=1

    print('>>pos:neg: ', pos_size, neg_size)
    return examples


def get_CNN_DailyMail_examples(filename):
    #/export/home/Dataset/para_entail_datasets/DUC/train_in_entail.txt
    print('loading...', filename)
    readfile = codecs.open(filename, 'r', 'utf-8')
    start = False
    examples = []
    guid_id = -1
    pos_size = 0
    neg_size = 0
    for line in readfile:
        if len(line.strip()) == 0:
            start = False
        else:
            parts = line.strip().split('\t')
            if parts[0] == 'document>>':
                start = True
                premise = parts[1].strip()
            elif parts[0] == 'positive>>':
                guid_id+=1
                pos_hypo = parts[1].strip()
                examples.append(InputExample(guid=str(guid_id), text_a=premise, text_b=pos_hypo, label='entailment'))
                pos_size+=1
            elif parts[0] == 'negative>>' and parts[1] != '#ShuffleWord#>>' and parts[1] != '#RemoveWord#>>':
                guid_id+=1
                neg_hypo = parts[2].strip()

                # if filename.find('train_in_entail') > -1:
                examples.append(InputExample(guid=str(guid_id), text_a=premise, text_b=neg_hypo, label='not_entailment'))
                neg_size+=1
                # else:
                #     rand_prob = random.uniform(0, 1)
                #     if rand_prob > 3/4:
                #         examples.append(InputExample(guid=str(guid_id), text_a=premise, text_b=neg_hypo, label='not_entailment'))
                #         neg_size+=1

    print('>>pos:neg: ', pos_size, neg_size)
    return examples

def get_MCTest_examples():


def load_data():
    '''DUC'''
