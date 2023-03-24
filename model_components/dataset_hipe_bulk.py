import pandas as pd
from torch.utils.data.dataloader import DataLoader
from ast import literal_eval
import torch
from transformers import AutoTokenizer, BertModel
from model_config.ner_tr import NerTr

class TextDatasetBulk(torch.utils.data.Dataset):
    def __init__(self, csv, window_len, step_len, device, tokenizer_name, max_len_tokens):
        super(TextDatasetBulk, self).__init__()
        self.words = [item for sublist in csv['words'] for item in literal_eval(sublist)]
        self.ner_c = [item for sublist in csv['ner_c'] for item in literal_eval(sublist)]
        self.ner_f = [item for sublist in csv['ner_f'] for item in literal_eval(sublist)]
        self.window_len = window_len
        self.step_len = step_len
        self.device = device
        self.max_len_tokens = max_len_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.words_bulk, self.ner_c_bulk, self.ner_f_bulk = self.bulk_data()

    def split_list(self, lst, padding):
        # 列表长度
        length = len(lst)

        # 计算切割窗口的数量
        num_windows = (length - self.window_len) // self.step_len + 1

        # 如果列表不足一个切割窗口，则在列表末尾填充零
        if length < self.window_len:
            lst += [padding] * (self.window_len - length)
            return [lst]

        # 划分列表
        windows = [lst[i * self.step_len:i * self.step_len + self.window_len]
                   for i in range(num_windows)]

        # 如果最后一个窗口不足 m 个元素，则在窗口末尾填充零
        if len(windows[-1]) < self.window_len:
            windows[-1] += [padding] * (self.window_len - len(windows[-1]))

        return windows

    def bulk_data(self):
        words_bulk = self.split_list(self.words, padding='.')
        ner_c_bulk = self.split_list(self.ner_c, padding=0)
        ner_f_bulk = self.split_list(self.ner_f, padding=8)
        return words_bulk, ner_c_bulk, ner_f_bulk

    def __len__(self):
        return len(self.words_bulk)

    def __getitem__(self, item):
        output_tokenizer = self.tokenizer(self.words_bulk[item], is_split_into_words=True,
                                   padding="max_length", max_length=self.max_len_tokens)
        input_ids = output_tokenizer['input_ids']
        attention_mask_bert = output_tokenizer['attention_mask']
        word_ids = [-100 if element is None
                               else element for element in output_tokenizer.word_ids()]
        return {
            'input_ids': torch.tensor(input_ids).to(self.device),
            'words_ids': torch.tensor(word_ids).to(self.device),
            'attention_mask_bert': torch.tensor(attention_mask_bert).to(self.device),
            'label_croase': torch.tensor(self.ner_c_bulk[item]).to(self.device),
            'label_fine': torch.tensor(self.ner_f_bulk[item]).to(self.device)
        }

# if __name__ == "__main__":
#     csv = pd.read_csv('../data/train_fr.csv')
#     dataset = TextDatasetBulk(csv, 10, 5, 'cuda:0', 'camembert-base', 50)
#     dataloader = DataLoader(dataset, batch_size=4)
#     bert = BertModel.from_pretrained('camembert-base')
#     model = NerTr(bert_model=bert, sim_dim=768, max_len_words=10,
#                   num_ner=10, ann_type='croase', device='cuda:0')
#     model.to('cuda:0')
#     for _, data in enumerate(dataloader):
#         print(data)
#         model(data)



