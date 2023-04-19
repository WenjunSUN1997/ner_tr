import torch
from torch.utils.data.dataloader import DataLoader
from model_components.dataset_hipe import TextDataset
import pandas as pd
from transformers import BertModel, LlamaTokenizer, AutoTokenizer
from tqdm import tqdm
from datasets import load_dataset
from tqdm import tqdm

def data_preprocess(data, bert, max_len_token=618):
    '''
    :param data: output of dataloader 'input_ids'[bs,1000], 'input_ids_length'[bs,618]
                'attention_mask'bs.618, 'label_croase' 'label_fine'
    :return:
    '''
    output_bert = bert(input_ids=data['input_ids'],
                       attention_mask=data['attention_mask_bert'])
    last_hidden_state = output_bert['last_hidden_state']
    b_s, true_length, _ = last_hidden_state.shape
    bert_avg = []

    for b_s_index in range(b_s):
        last_hidden_state_one_batch = last_hidden_state[b_s_index]
        last_hidden_state_real = last_hidden_state_one_batch[
                                 :len(data['input_ids_length'][b_s_index])]
        temp = []
        i = 0
        for length in data['input_ids_length'][b_s_index][
            data['input_ids_length'][b_s_index].nonzero().squeeze()]:
            temp.append(torch.mean(last_hidden_state_real[i:i + length], dim=0))
            i += length
        stack = torch.stack(temp, dim=0)
        padding = (0, 0, 0, max_len_token-len(stack))
        bert_avg.append(torch.nn.functional.pad(stack, padding))
# {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3,
# 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7,
# 'I-MISC': 8}

def data_augmentation():
    tag_dict = {'O': 0, 'B-PER': 1, 'I-PER': 2,
                'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5,
                'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
    words = []
    ner_c = []
    ner_f = []
    with open('../data/wikiner_fr', 'r', encoding="utf8") as file:
        text_1 = file.read()
    # with open('../data/wikiner_en_2', 'r', encoding="utf8") as file:
    #     text_2 = file.read()

    text = text_1
    text = text.replace('\n', ' ')
    text = text.split(' ')
    text = [x for x in text if x != '']
    for text_cell in tqdm(text):
        # print(text_cell)
        token, _, tag = text_cell.split('|')
        words.append(token)
        ner_c.append(tag_dict[tag])
        ner_f.append(tag_dict[tag])

    df = pd.read_csv('../data/train_fr.csv')
    a = df.append({'words': words, 'ner_c': ner_c, 'ner_f': ner_f}, ignore_index=True)
    a.reset_index(drop=True)
    a.to_csv('../data/train_fr_augmentation.csv')





if __name__ == "__main__":
    data_augmentation()

    # dataset = load_dataset('conll2002')
    # df = dataset['train']
    # df = pd.DataFrame(df)
    # words = [x for x in df['tokens']]
    # ner_c = [x for x in df['ner_tags']]
    # ner_f = [x for x in df['ner_tags']]
    # d = pd.DataFrame({'words': words, 'ner_c':ner_c, 'ner_f':ner_f})
    # d.to_csv('../data/test_conll2003.csv')