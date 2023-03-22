import pandas as pd
from ast import literal_eval
from itertools import islice

def split_list(lst, m, n, padding):
    # 列表长度
    length = len(lst)

    # 计算切割窗口的数量
    num_windows = (length - m) // n + 1

    # 如果列表不足一个切割窗口，则在列表末尾填充零
    if length < m:
        lst += [padding] * (m - length)
        return [lst]

    # 划分列表
    windows = [lst[i * n:i * n + m] for i in range(num_windows)]

    # 如果最后一个窗口不足 m 个元素，则在窗口末尾填充零
    if len(windows[-1]) < m:
        windows[-1] += [padding] * (m - len(windows[-1]))

    return windows

def sliding_window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield

df = pd.read_csv('../data/train_fr.csv')
words_bulk = []
ner_c_bulk = []
ner_f_bulk = []
words = df['words']
ner_c = df['ner_c']
ner_f = df['ner_f']
window_len = 10
step_len = 5

for index in range(len(words)):
    words_bulk += split_list(literal_eval(words[index]), window_len, step_len, padding='.')
    ner_c_bulk += split_list(literal_eval(ner_c[index]), window_len, step_len, padding=0)
    ner_f_bulk += split_list(literal_eval(ner_f[index]), window_len, step_len, padding=8)
    print(words_bulk)
df = {'words':words_bulk,
      'ner_c':ner_c_bulk,
      'ner_f':ner_f_bulk}
df = pd.DataFrame(df)
df.to_csv('../data/train_fr_bulk.csv')
