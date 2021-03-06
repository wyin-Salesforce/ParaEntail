import json_lines
import codecs
from transformers.data.processors.utils import InputExample

def get_DUC_examples(prefix, hypo_only=False):
    #/export/home/Dataset/para_entail_datasets/DUC/train_in_entail.txt
    path = '/export/home/Dataset/para_entail_datasets/DUC/'
    # filename = path+prefix+'_in_entail.txt'
    filename = path+prefix+'_in_entail.harsh.txt'
    print('loading DUC...', filename)
    readfile = codecs.open(filename, 'r', 'utf-8')
    start = False
    examples = []
    extra_labels = []
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
                if len(premise) == 0 or len(pos_hypo)==0:
                    # print('DUC premise:', premise)
                    # print('hypothesis:', pos_hypo)
                    continue

                if hypo_only:
                    examples.append(InputExample(guid=str(guid_id), text_a=pos_hypo, text_b=None, label='entailment'))
                    extra_labels.append('entailment')
                else:
                    examples.append(InputExample(guid=str(guid_id), text_a=premise, text_b=pos_hypo, label='entailment'))
                    extra_labels.append('entailment')
                pos_size+=1
            elif parts[0] == 'negative>>' and parts[1] != '#ShuffleWord#>>' and parts[1] != '#RemoveWord#>>':
                guid_id+=1
                neg_hypo = parts[2].strip()
                if len(premise) == 0 or len(neg_hypo)==0:
                    # print('DUC premise:', premise)
                    # print('hypothesis:', neg_hypo)
                    continue

                if hypo_only:
                    examples.append(InputExample(guid=str(guid_id), text_a=neg_hypo, text_b=None, label='not_entailment'))
                    extra_labels.append(parts[1])
                else:
                    examples.append(InputExample(guid=str(guid_id), text_a=premise, text_b=neg_hypo, label='not_entailment'))
                    extra_labels.append(parts[1])
                neg_size+=1
                # 
                # examples.append(InputExample(guid=str(guid_id), text_a=neg_hypo, text_b=pos_hypo, label='not_entailment'))
                # examples.append(InputExample(guid=str(guid_id), text_a=neg_hypo, text_b=neg_hypo, label='entailment'))

    print('>>pos:neg: ', pos_size, neg_size)
    print('DUC size:', len(examples))
    return examples, extra_labels, pos_size

def get_Curation_examples(prefix, hypo_only=False):
    #/export/home/Dataset/para_entail_datasets/DUC/train_in_entail.txt
    path = '/export/home/Dataset/para_entail_datasets/Curation/'
    filename = path+prefix+'_in_entail.txt'
    print('loading Curation...', filename)
    readfile = codecs.open(filename, 'r', 'utf-8')
    start = False
    examples = []
    extra_labels = []
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
                if len(premise) == 0 or len(pos_hypo)==0:
                    continue
                if hypo_only:
                    examples.append(InputExample(guid=str(guid_id), text_a=pos_hypo, text_b=None, label='entailment'))
                    extra_labels.append('entailment')
                else:
                    examples.append(InputExample(guid=str(guid_id), text_a=premise, text_b=pos_hypo, label='entailment'))
                    extra_labels.append('entailment')
                pos_size+=1
            elif parts[0] == 'negative>>' and parts[1] != '#ShuffleWord#>>' and parts[1] != '#RemoveWord#>>':
                guid_id+=1
                neg_hypo = parts[2].strip()
                if len(premise) == 0 or len(neg_hypo)==0:
                    continue

                if hypo_only:
                    examples.append(InputExample(guid=str(guid_id), text_a=neg_hypo, text_b=None, label='not_entailment'))
                    extra_labels.append(parts[1])
                else:
                    examples.append(InputExample(guid=str(guid_id), text_a=premise, text_b=neg_hypo, label='not_entailment'))
                    extra_labels.append(parts[1])
                neg_size+=1

    print('>>pos:neg: ', pos_size, neg_size)
    print('Curation size:', len(examples))
    return examples,extra_labels,  pos_size

def get_CNN_DailyMail_examples(prefix, hypo_only=False):
    #/export/home/Dataset/para_entail_datasets/DUC/train_in_entail.txt
    path = '/export/home/Dataset/para_entail_datasets/CNN_DailyMail/'
    filename = path+prefix+'_in_entail.txt'
    print('loading CNN_DailyMail...', filename)
    readfile = codecs.open(filename, 'r', 'utf-8')
    start = False
    examples = []
    extra_labels = []
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
                if len(premise) == 0 or len(pos_hypo)==0:
                    # print('CNN premise:', premise)
                    # print('hypothesis:', pos_hypo)
                    continue
                if hypo_only:
                    examples.append(InputExample(guid=str(guid_id), text_a=pos_hypo, text_b=None, label='entailment'))
                else:
                    examples.append(InputExample(guid=str(guid_id), text_a=premise, text_b=pos_hypo, label='entailment'))

                extra_labels.append('entailment')

                pos_size+=1
            elif parts[0] == 'negative>>' and parts[1] != '#ShuffleWord#>>' and parts[1] != '#RemoveWord#>>':
                guid_id+=1
                neg_hypo = parts[2].strip()

                # if filename.find('train_in_entail') > -1:
                if len(premise) == 0 or len(neg_hypo)==0:
                    # print('CNN premise:', premise)
                    # print('neg_hypo:', neg_hypo)
                    continue

                if hypo_only:
                    examples.append(InputExample(guid=str(guid_id), text_a=neg_hypo, text_b=None, label='not_entailment'))
                else:
                    examples.append(InputExample(guid=str(guid_id), text_a=premise, text_b=neg_hypo, label='not_entailment'))
                extra_labels.append(parts[1])
                neg_size+=1
                # else:
                #     rand_prob = random.uniform(0, 1)
                #     if rand_prob > 3/4:
                #         examples.append(InputExample(guid=str(guid_id), text_a=premise, text_b=neg_hypo, label='not_entailment'))
                #         neg_size+=1

    print('>>pos:neg: ', pos_size, neg_size)
    print('CNN size:', len(examples))
    return examples, extra_labels, pos_size

def get_MCTest_examples(prefix, hypo_only=False):
    path = '/export/home/Dataset/para_entail_datasets/MCTest/'
    filename = path+prefix+'_in_entail.txt'
    print('loading MCTest...', filename)
    readfile = codecs.open(filename, 'r', 'utf-8')
    guid_id = 0
    pos_size = 0
    examples = []
    extra_labels = []
    for line in readfile:
        guid_id+=1
        parts = line.strip().split('\t')
        if len(parts) ==3:
            premise = parts[1]
            hypothesis = parts[2]
            label = 'entailment' if parts[0] == 'entailment' else 'not_entailment'
            if label == 'entailment':
                pos_size+=1
            if len(premise) == 0 or len(hypothesis)==0:
                # print('MCTest premise:', premise)
                # print('hypothesis:', hypothesis)
                continue

            if hypo_only:
                examples.append(InputExample(guid=prefix+str(guid_id), text_a=hypothesis, text_b=None, label=label))
            else:
                examples.append(InputExample(guid=prefix+str(guid_id), text_a=premise, text_b=hypothesis, label=label))
            extra_labels.append(label)
    print('MCTest size:', len(examples))
    return examples, extra_labels, pos_size

def get_FEVER_examples(prefix, hypo_only=False):
    '''
    train_fitems.jsonl, dev_fitems.jsonl, test_fitems.jsonl
    dev_fitems.label.recovered.jsonl
    '''
    examples = []
    path = '/export/home/Dataset/para_entail_datasets/nli_FEVER/nli_fever/'
    filename = path+prefix+'_fitems.jsonl'
    if prefix == 'test':
        filename = path+'dev_fitems.label.recovered.jsonl'
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
            if len(premise) == 0 or len(hypothesis)==0:
                # print('FEVER premise:', premise)
                # print('hypothesis:', hypothesis)
                # print(line)
                # exit(0)
                continue

            if hypo_only:
                examples.append(InputExample(guid=str(guid_id), text_a=hypothesis, text_b=None, label=label))
            else:
                examples.append(InputExample(guid=str(guid_id), text_a=premise, text_b=hypothesis, label=label))
    print('FEVER size:', len(examples))
    return examples, pos_size

def get_ANLI_examples(prefix, hypo_only=False):
    folders = ['R1', 'R2', 'R3']
    examples = []
    extra_labels = []
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
                if len(premise) == 0 or len(hypothesis)==0:
                    # print('ANLI premise:', premise)
                    # print('hypothesis:', hypothesis)
                    continue
                if hypo_only:
                    examples.append(InputExample(guid=str(guid_id), text_a=hypothesis, text_b=None, label=label))
                else:
                    examples.append(InputExample(guid=str(guid_id), text_a=premise, text_b=hypothesis, label=label))
                extra_labels.append(label)
    print('ANLI size:', len(examples))
    return examples, extra_labels, pos_size




def load_train_data(hypo_only=False):

    '''DUC'''
    duc_examples, duc_extra_labels, duc_pos_size = get_DUC_examples('train', hypo_only=hypo_only)
    '''CNN'''
    cnn_examples, cnn_extra_labels, cnn_pos_size = get_CNN_DailyMail_examples('train', hypo_only=hypo_only)
    '''MCTest'''
    mctest_examples, mctest_extra_labels, mctest_pos_size = get_MCTest_examples('train', hypo_only=hypo_only)
    '''Curation'''
    curation_examples, curation_extra_labels, curation_pos_size = get_Curation_examples('train', hypo_only=hypo_only)
    '''ANLI'''
    anli_examples, anli_extra_labels, anli_pos_size = get_ANLI_examples('train', hypo_only=hypo_only)

    # print('duc_examples size:', len(duc_examples))
    # print('cnn_examples size:', len(cnn_examples))
    train_examples = (
                        duc_examples+
                        cnn_examples+
                        mctest_examples+
                        curation_examples+
                        anli_examples
                        )

    train_extra_labels = (
                        duc_extra_labels+
                        cnn_extra_labels+
                        mctest_extra_labels+
                        curation_extra_labels+
                        anli_extra_labels
                        )
    pos_size = (
                duc_pos_size+
                cnn_pos_size+
                mctest_pos_size+
                curation_pos_size+
                anli_pos_size
                )
    print('train size:', len(train_examples), ' pos size:', pos_size, ' ratio:', pos_size/len(train_examples))

    return train_examples, train_extra_labels


def load_dev_data(hypo_only=False):
    '''test size: 125646  pos size: 14309; 11.38%'''
    '''DUC'''
    duc_examples, duc_extra_labels, duc_pos_size = get_DUC_examples('dev', hypo_only=hypo_only)
    '''CNN'''
    cnn_examples, cnn_extra_labels, cnn_pos_size = get_CNN_DailyMail_examples('dev', hypo_only=hypo_only)
    '''MCTest'''
    mctest_examples, mctest_extra_labels, mctest_pos_size = get_MCTest_examples('dev', hypo_only=hypo_only)
    '''Curation'''
    curation_examples, curation_extra_labels, curation_pos_size = get_Curation_examples('dev', hypo_only=hypo_only)
    '''ANLI'''
    anli_examples, anli_extra_labels, anli_pos_size = get_ANLI_examples('dev', hypo_only=hypo_only)

    dev_examples = (
                        duc_examples+
                        cnn_examples+
                        mctest_examples+
                        curation_examples+
                        anli_examples
                        )
    dev_extra_labels = (
                        duc_extra_labels+
                        cnn_extra_labels+
                        mctest_extra_labels+
                        curation_extra_labels+
                        anli_extra_labels
                        )
    pos_size = (
                duc_pos_size+
                cnn_pos_size+
                mctest_pos_size+
                curation_pos_size+
                anli_pos_size
                )

    print('dev size:', len(dev_examples), ' pos size:', pos_size, ' ratio:', pos_size/len(dev_examples))
    return dev_examples, dev_extra_labels


def load_test_data(hypo_only=False):
    '''test size: 125646  pos size: 14309; 11.38%'''
    '''DUC'''
    duc_examples, duc_extra_labels, duc_pos_size = get_DUC_examples('test', hypo_only=hypo_only)
    '''CNN'''
    # cnn_examples, cnn_extra_labels, cnn_pos_size = get_CNN_DailyMail_examples('test', hypo_only=hypo_only)
    # '''MCTest'''
    # mctest_examples, mctest_extra_labels, mctest_pos_size = get_MCTest_examples('test', hypo_only=hypo_only)
    # '''Curation'''
    # curation_examples, curation_extra_labels, curation_pos_size = get_Curation_examples('test', hypo_only=hypo_only)
    # '''ANLI'''
    # anli_examples, anli_extra_labels, anli_pos_size = get_ANLI_examples('test', hypo_only=hypo_only)

    test_examples = (
                        duc_examples
                        # cnn_examples+
                        # mctest_examples+
                        # curation_examples+
                        # anli_examples
                        )
    test_extra_labels = (
                        duc_extra_labels
                        # cnn_extra_labels+
                        # mctest_extra_labels+
                        # curation_extra_labels+
                        # anli_extra_labels
                        )
    pos_size = (
                duc_pos_size
                # cnn_pos_size+
                # mctest_pos_size+
                # curation_pos_size+
                # anli_pos_size
                )

    print('test size:', len(test_examples), ' pos size:', pos_size, ' ratio:', pos_size/len(test_examples))
    return test_examples, test_extra_labels

if __name__ == "__main__":
    # load_train_data()
    load_test_data()

    '''
    train size: 1404446  pos size: 304352
    test size: 123857  pos size: 20975
    '''
