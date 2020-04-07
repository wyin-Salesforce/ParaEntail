import json_lines


def get_DUC_examples(prefix):
    #/export/home/Dataset/para_entail_datasets/DUC/train_in_entail.txt
    path = '/export/home/Dataset/para_entail_datasets/DUC/'
    filename = path+prefix+'_in_entail.txt'
    print('loading DUC...', filename)
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
                examples.append(InputExample(guid=str(guid_id), text_a=premise, text_b=neg_hypo, label='not_entailment'))
                neg_size+=1
                # if filename.find('train_in_entail') > -1:
                #     examples.append(InputExample(guid=str(guid_id), text_a=premise, text_b=neg_hypo, label='not_entailment'))
                #     neg_size+=1
                # else:
                #     rand_prob = random.uniform(0, 1)
                #     if rand_prob > 3/4:
                #         examples.append(InputExample(guid=str(guid_id), text_a=premise, text_b=neg_hypo, label='not_entailment'))
                #         neg_size+=1

    print('>>pos:neg: ', pos_size, neg_size)
    return examples


def get_CNN_DailyMail_examples(prefix):
    #/export/home/Dataset/para_entail_datasets/DUC/train_in_entail.txt
    path = '/export/home/Dataset/para_entail_datasets/CNN_DailyMail/'
    filename = path+prefix+'_in_entail.txt'
    print('loading CNN_DailyMail...', filename)
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

def get_MCTest_examples(prefix):
    path = '/export/home/Dataset/para_entail_datasets/MCTest/'
    readfile = codecs.open(path+prefix+'_in_entail.txt', 'r', 'utf-8')
    guid_id = 0
    examples = []
    for line in readfile:
        guid_id+=1
        parts = line.strip().split('\t')
        if len(parts) ==3:
            premise = parts[1]
            hypothesis = parts[2]
            label = parts[0]
            examples.append(InputExample(guid=prefix+str(guid_id), text_a=premise, text_b=hypothesis, label=label))
    return examples

def get_FEVER_examples(prefix):
    '''
    train_fitems.jsonl, dev_fitems.jsonl, test_fitems.jsonl
    '''
    examples = []
    path = '/export/home/Dataset/para_entail_datasets/nli_FEVER/nli_fever/'
    filename = path+prefix+'_fitems.jsonl'
    guid_id = 0
    with open(filename, 'r') as f:
        for line in json_lines.reader(f):
            guid_id+=1
            premise = line.get('context')
            hypothesis = line.get('query')
            label = 'entailment' if line.get('label') == 'SUPPORTS' else 'non_entailment'
            examples.append(InputExample(guid=str(guid_id), text_a=premise, text_b=hypothesis, label=label))
    return examples

def get_ANLI_examples(prefix):
    folders = ['R1', 'R2', 'R3']
    examples = []
    guid_id = 0
    path = '/export/home/Dataset/para_entail_datasets/ANLI/anli_v0.1/'
    for folder in folders:
        with open(path+folder+'/'+prefix+'.jsonl', 'r') as f:
            for line in json_lines.reader(f):
                guid_id+=1
                premise = line.get('context')
                hypothesis = line.get('hypothesis')
                label = 'entailment' if line.get('label') == 'e' else 'non_entailment'
                examples.append(InputExample(guid=str(guid_id), text_a=premise, text_b=hypothesis, label=label))
    return examples




def load_train_data():
    '''DUC'''
    duc_examples = get_DUC_examples('train')
    '''CNN'''
    cnn_examples = get_CNN_DailyMail_examples('train')
    '''MCTest'''
    mctest_examples = get_MCTest_examples('train')
    '''FEVER'''
    fever_examples = get_FEVER_examples('train')
    '''ANLI'''
    anli_examples = get_ANLI_examples('train')

    train_examples = duc_examples+cnn_examples+mctest_examples+fever_examples+anli_examples
    print('train size:', len(train_examples))

if __name__ == "__main__":
    load_train_data()
