import json_lines
import codecs
from transformers.data.processors.utils import InputExample

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
    print('size:', len(examples))
    return examples, pos_size


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
    print('size:', len(examples))
    return examples, pos_size

def get_MCTest_examples(prefix):
    path = '/export/home/Dataset/para_entail_datasets/MCTest/'
    filename = path+prefix+'_in_entail.txt'
    print('loading MCTest...', filename)
    readfile = codecs.open(filename, 'r', 'utf-8')
    guid_id = 0
    pos_size = 0
    examples = []
    for line in readfile:
        guid_id+=1
        parts = line.strip().split('\t')
        if len(parts) ==3:
            premise = parts[1]
            hypothesis = parts[2]
            label = 'entailment' if parts[0] == 'entailment' else 'not_entailment'
            if label == 'entailment':
                pos_size+=1
            examples.append(InputExample(guid=prefix+str(guid_id), text_a=premise, text_b=hypothesis, label=label))
    print('size:', len(examples))
    return examples, pos_size

def get_FEVER_examples(prefix):
    '''
    train_fitems.jsonl, dev_fitems.jsonl, test_fitems.jsonl
    '''
    examples = []
    path = '/export/home/Dataset/para_entail_datasets/nli_FEVER/nli_fever/'
    filename = path+prefix+'_fitems.jsonl'
    print('loading FEVER...', filename)
    guid_id = 0
    pos_size = 0
    with open(filename, 'r') as f:
        for line in json_lines.reader(f):
            guid_id+=1
            premise = line.get('context')
            hypothesis = line.get('query')
            label = 'entailment' if line.get('label') == 'SUPPORTS' else 'not_entailment'
            if label == 'entailment':
                pos_size+=1
            examples.append(InputExample(guid=str(guid_id), text_a=premise, text_b=hypothesis, label=label))
    print('size:', len(examples))
    return examples, pos_size

def get_ANLI_examples(prefix):
    folders = ['R1', 'R2', 'R3']
    examples = []
    guid_id = 0
    pos_size = 0
    path = '/export/home/Dataset/para_entail_datasets/ANLI/anli_v0.1/'
    for folder in folders:
        filename = path+folder+'/'+prefix+'.jsonl'
        print('loading ANLI...', filename)
        with open(filename, 'r') as f:
            for line in json_lines.reader(f):
                guid_id+=1
                premise = line.get('context')
                hypothesis = line.get('hypothesis')
                label = 'entailment' if line.get('label') == 'e' else 'not_entailment'
                if label == 'entailment':
                    pos_size+=1
                examples.append(InputExample(guid=str(guid_id), text_a=premise, text_b=hypothesis, label=label))
    print('size:', len(examples))
    return examples, pos_size




def load_train_data():
    '''train size: 1120541  pos size: 269096; 24.01%'''
    '''DUC'''
    duc_examples, duc_pos_size = get_DUC_examples('train')
    '''CNN'''
    cnn_examples, cnn_pos_size = get_CNN_DailyMail_examples('train')
    '''MCTest'''
    mctest_examples, mctest_pos_size = get_MCTest_examples('train')
    '''FEVER'''
    fever_examples, fever_pos_size = get_FEVER_examples('train')
    '''ANLI'''
    anli_examples, anli_pos_size = get_ANLI_examples('train')

    train_examples = duc_examples+cnn_examples+mctest_examples+fever_examples+anli_examples
    pos_size = duc_pos_size+cnn_pos_size+mctest_pos_size+fever_pos_size+anli_pos_size
    print('train size:', len(train_examples), ' pos size:', pos_size)
    return train_examples

def load_test_data():
    '''test size: 125646  pos size: 14309; 11.38%'''
    '''DUC'''
    duc_examples, duc_pos_size = get_DUC_examples('test')
    '''CNN'''
    cnn_examples, cnn_pos_size = get_CNN_DailyMail_examples('test')
    '''MCTest'''
    mctest_examples, mctest_pos_size = get_MCTest_examples('test')
    '''FEVER'''
    fever_examples, fever_pos_size = get_FEVER_examples('test')
    '''ANLI'''
    anli_examples, anli_pos_size = get_ANLI_examples('test')

    test_examples = duc_examples+cnn_examples+mctest_examples+fever_examples+anli_examples
    pos_size = duc_pos_size+cnn_pos_size+mctest_pos_size+fever_pos_size+anli_pos_size
    print('test size:', len(test_examples), ' pos size:', pos_size)
    return test_examples

if __name__ == "__main__":
    # load_train_data()
    load_test_data()
