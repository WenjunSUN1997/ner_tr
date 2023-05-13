import pandas as pd
from ast import literal_eval
from collections import Counter

# label_list = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG',
#                   'B-LOC', 'I-LOC', 'B-HumanProd', 'I-HumanProd']
label_list = ['O','B-prod','I-prod','B-loc','I-loc','B-org','I-org',
                  'B-time','I-time','B-pers','I-pers']
df = pd.read_csv('../data/dev_20_en.csv')
a = []

for x in df['ner_c']:
    a += literal_eval(x)

counter = Counter(a)

for item, count in counter.items():
    print(f"{label_list[item]} &{count}  \\\\")



