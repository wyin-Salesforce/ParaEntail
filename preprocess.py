
import codecs

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
    readfile = codecs.open(fil, 'r', 'utf-8')
    pairlist = []
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
        if line.strip().startswith('</SUM>'):
            summary_start = False
            pairlist.append((doc_id, summary))
        if summary_start:
            summary +=' '+line.strip()
    print('size:', len(pairlist))
    print(pairlist)

def load_DUC():
    #DUC2001
    folder_namelist = ['d01a','d02a','d03a','d07b','d09b','d10b','d16c','d17c','d18c','d20d','d21d',
    'd23d','d25e','d26e','d29e','d33f','d35f','d36f','d38g','d40g','d42g','d46h','d47h','d48h','d49i',
    'd51i','d52i','d55k','d58k','d60k']

    for foldername in folder_namelist:
        last_char = foldername[-1]
        subfolder = foldername+last_char
        docsfolder = 'docs'
        summary_read = codecs.open('/export/home/Dataset/para_entail_datasets/DUC/DUC_data/data/duc01/data/training/'+foldername+'/'+subfolder+'/perdocs', 'r', 'utf-8')


if __name__ == "__main__":
    load_per_docs_file('/export/home/Dataset/para_entail_datasets/DUC/DUC_data/data/duc01/data/training/d49i/d49ii/perdocs')
