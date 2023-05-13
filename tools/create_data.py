import pandas as pd
from datasets.features import ClassLabel
import pyarrow as pa
import re
from datasets import Dataset
# ['O', 'B-ORG', 'I-ORG', 'B-LOC',
# 'I-LOC', 'B-PER', 'I-PER',
# 'B-HumanProd', 'I-HumanProd', '_']

def get_label():
    label_set = df['NE-COARSE-LIT'].unique()
    # define the label mapping for NER
    label_list = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG',
                  'B-LOC', 'I-LOC', 'B-HumanProd', 'I-HumanProd']
    label_list = ['O','B-pers','I-pers','B-work','I-work','B-scope','I-scope',
                  'B-loc', 'B-date','I-date', ]
    label_list = ['O','B-LOC','I-LOC','B-BUILDING','I-BUILDING','B-STREET','I-STREET']
    label_list = ['O','B-prod','I-prod','B-loc','I-loc','B-org','I-org',
                  'B-time','I-time','B-pers','I-pers']
    # label_list = ['O','B-pers','I-pers',
    #               'B-work','I-work',
    #               'B-scope','I-scope',
    #               'B-loc','I-loc',
    #               'B-date','I-date',
    #               'B-object','I-object']

    label_num = len(label_list)
    labels = ClassLabel(num_classes=label_num, names=label_list)
    return labels, label_num

def get_label_fine():
    label_set_fin = df['NE-FINE-LIT'].unique()
    label_list_fin = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG',
                      'B-LOC', 'I-LOC', 'B-HumanProd', 'I-HumanProd',
                      'B-PER.author', 'I-PER.author']
    label_list_fin = ['O','B-pers.author','I-pers.author',
                      'B-work.journal','I-work.journal',
                      'B-scope','I-scope',
                      'B-pers.editor','I-pers.editor',
                      'B-loc','I-loc',
                      'B-date','I-date',
                      'B-object.manuscr','I-object.manuscr',
                     'B-pers.other','I-pers.other',
                      'B-work.primlit','I-work.primlit',
                      'B-pers.myth','I-pers.myth',
                      'B-work.seclit','I-work.seclit',
                      'B-work.other','I-work.other',
                      'B-work.fragm','I-work.fragm']

    # label_set_fin = ['O','B-LOC','I-LOC','B-BUILDING','I-BUILDING','B-STREET','I-STREET']


    label_num_fin = len(label_list_fin)
    labels_fin = ClassLabel(num_classes=label_num_fin, names=label_list_fin)
    return labels_fin, label_num_fin

def simple_preprocess(dataframe):
    # Add end_of_document token in df
    dataframe = dataframe.dropna(subset=['TOKEN'])

    # Filter out metadata rows beginning with #
    dataframe = dataframe[~dataframe['TOKEN'].astype(str).str.startswith('#')]
    dataframe = dataframe[~dataframe['TOKEN'].astype(str).str.startswith('\t')]

    #transforming nan var from Float to string to use in (***)
    dataframe.MISC = dataframe.MISC.fillna('')

    return dataframe

def create_huggingface_file(dataframe):
    #creating dataset in json
    hug_out = []
    idx = 0
    items = {'id': idx, 'words': [], 'ner': []}
    hug_out.append(items)
    for index, row in dataframe.iterrows():
        if not re.search(r'EndOfSentence', row['MISC']):
            items['words'].append(row['TOKEN'])
            try:
                items['ner'].append(labels.str2int(row['NE-COARSE-LIT']))
            except:
                items['ner'].append(0)

        else:
            items['words'].append(row['TOKEN'])
            try:
                items['ner'].append(labels.str2int(row['NE-COARSE-LIT']))
            except:
                items['ner'].append(0)
            idx += 1
            items = {'id': idx,'words':[ ], 'ner': [ ]}
            hug_out.append(items)
    #filter hug_out out, delete items which has len(words) > 380
    #hug_out = filter(lambda x: len(x['words']) < 380, hug_out)
    #json to df
    hug_out = pd.DataFrame(hug_out)

    # delete all sentences that are too long
    #hug_out = hug_out[hug_out['words'].map(len) < 512] #why does not work? QA

    ### convert to Huggingface dataset
    hug_out = Dataset(pa.Table.from_pandas(hug_out))

    return hug_out

def create_huggingface_file_fine(dataframe):
    hug_out = []
    idx = 0
    items = {'id': idx,'words': [], 'ner_fin': []}
    hug_out.append(items)
    for index, row in dataframe.iterrows():
        if  not re.search(r'EndOfSentence', row['MISC']):
            items['words'].append(row['TOKEN'])
            if row['NE-FINE-LIT']!='O':
                items['ner_fin'].append(labels_fin.str2int(row['NE-FINE-LIT']))
            else:
                items['ner_fin'].append(labels_fin.str2int(row['NE-COARSE-LIT']))

        else:
            items['words'].append(row['TOKEN'])
            if row['NE-FINE-LIT']!='O':
                items['ner_fin'].append(labels_fin.str2int(row['NE-FINE-LIT']))
            else:
                items['ner_fin'].append(labels_fin.str2int(row['NE-COARSE-LIT']))
            idx += 1
            items = {'id': idx,'words':[ ], 'ner_fin': [ ]}
            hug_out.append(items)

    hug_out = pd.DataFrame(hug_out)
    hug_out = Dataset(pa.Table.from_pandas(hug_out))

    return hug_out

path = '../data/HIPE-2022-v2.0-hipe2020-dev-en.tsv'
df = pd.read_csv(path, sep='\t', skip_blank_lines=False, engine='python', quoting=3)
df = simple_preprocess(df)
print(df)
# df.to_csv('../data/gt_ajmc_en.tsv', index=False)

labels, label_num = get_label()
labels_fin, label_num_fin =get_label_fine()

data_croase = create_huggingface_file(df)
# data_fine = create_huggingface_file_fine(df)
words = [v['words'] for v in data_croase]
ner_croase = [v['ner'] for v in data_croase ]
# ner_fine = [v['ner_fin'] for v in data_fine ]
dict_result = {'words': words, 'ner_c':ner_croase, 'ner_f':ner_croase}
data = pd.DataFrame(dict_result)
data.to_csv('../data/dev_20_en.csv')
print(len(data))