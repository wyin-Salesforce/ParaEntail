
import codecs
import import xmltodict
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
