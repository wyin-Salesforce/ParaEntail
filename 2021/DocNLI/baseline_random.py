import codecs
import json

readfile = codecs.open('/export/home/Dataset/para_entail_datasets/test.json', 'r', 'utf-8')

data = json.load(readfile)
print('len of data:', len(data))
for dic in data:
    print(dic.get('premise'))
    print(dic.get('label'))
    exit(0)
