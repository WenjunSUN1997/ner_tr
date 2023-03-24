import pandas as pd
from ast import literal_eval
import torch
from transformers import AutoTokenizer, BertModel, BertConfig
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from model_config.ner_tr import NerTr

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, csv, max_len_words, device,
                 max_len_tokens=200, tokenizer_name='camembert-base',
                 ):
        self.csv = csv
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len_words = max_len_words
        self.max_len_tokens = max_len_tokens
        self.device = device

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, item):
        words = literal_eval(self.csv['words'][item].lower())
        input_ids = [v[1:-1] for v in self.tokenizer(words)['input_ids']]
        input_ids_length = [len(v) for v in input_ids]
        input_ids_length = torch.tensor(input_ids_length
                                        + [0]*(self.max_len_words-len(input_ids_length)))
        attention_mask = torch.tensor([False]*len(input_ids)
                                      + [True]*(self.max_len_words-len(input_ids)))
        crf_mask = torch.tensor([True]*len(input_ids)
                                      + [False]*(self.max_len_words-len(input_ids)))
        label_croase = literal_eval(self.csv['ner_c'][item])
        label_croase = torch.tensor(label_croase + [-1]*(self.max_len_words-len(label_croase)))
        label_fine = literal_eval(self.csv['ner_f'][item])
        label_fine = torch.tensor(label_fine + [-1] * (self.max_len_words - len(label_fine)))
        input_ids = [item for sublist in input_ids for item in sublist]
        attention_mask_bert = torch.tensor([1] * len(input_ids) +
                                           [0] * (self.max_len_tokens - len(input_ids)))
        input_ids = torch.tensor(input_ids + [1] * (self.max_len_tokens - len(input_ids)))
        return {'input_ids': input_ids.to(self.device),
                'attention_mask_bert':attention_mask_bert.to(self.device),
                'input_ids_length': input_ids_length.to(self.device),
                'attention_mask': attention_mask.to(self.device),
                'crf_mask': crf_mask.to(self.device),
                'label_croase': label_croase.to(self.device),
                'label_fine': label_fine.to(self.device)}

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained('camembert-base')
    max = 0
    index = 0
    csv = pd.read_csv('../data/train_fr_bulk.csv')
    config = BertConfig.from_pretrained('camembert-base')
    config.max_position_embeddings = 1000
    bert = BertModel.from_pretrained('camembert-base', config=config)
    model = NerTr(bert_model=bert, sim_dim=768, max_len_words=618,
                  num_ner=10, ann_type='croase', device='cuda:0')

    dataset =TextDataset(csv, max_len_words=618, device='cuda:0')

    dataloader = DataLoader(dataset, batch_size=16)
    for data in tqdm(dataloader):
        model(data)