import pandas as pd
from torch.utils.data.dataloader import DataLoader
from ast import literal_eval
import torch
from transformers import AutoTokenizer, LlamaTokenizerFast, LlamaModel
from model_config.ner_tr import NerTr
from math import ceil

class TextDatasetBulk(torch.utils.data.Dataset):
    def __init__(self, csv, window_len, step_len, device,
                 tokenizer_name, max_len_tokens, goal, model_type):
        super(TextDatasetBulk, self).__init__()
        self.model_type = model_type
        self.words = [item for sublist in csv['words'] for item in literal_eval(sublist)]
        self.ner_c = [item for sublist in csv['ner_c'] for item in literal_eval(sublist)]
        self.ner_f = [item for sublist in csv['ner_f'] for item in literal_eval(sublist)]
        self.window_len = window_len
        self.goal = goal
        self.step_len = step_len
        self.device = device
        self.max_len_tokens = max_len_tokens
        if 'llama' not in tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            self.tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer_name)
            self.tokenizer.add_special_tokens({'pad_token': 'no'})
        self.words_bulk, self.ner_c_bulk, self.ner_f_bulk = self.bulk_data()

    def split_list(self, lst, padding):
        # 列表长度
        num_sub_lists = ceil(len(lst) / self.step_len)
        sub_lists = []
        for i in range(num_sub_lists):
            start = i * self.step_len
            end = min(start + self.window_len, len(lst))
            sub_list = lst[start:end]
            if len(sub_list) < self.window_len:
                sub_list += [padding] * (self.window_len - len(sub_list))
            sub_lists.append(sub_list)
        return sub_lists

    def bulk_data(self):
        words_bulk = self.split_list(self.words, padding='.')
        ner_c_bulk = self.split_list(self.ner_c, padding=0)
        ner_f_bulk = self.split_list(self.ner_f, padding=0)
        if self.goal == 'train':
            del_index = []
            words_bulk = [words_bulk[i] for i in range(len(words_bulk))
                          if i not in del_index]
            ner_c_bulk = [ner_c_bulk[i] for i in range(len(ner_c_bulk))
                          if i not in del_index]
            ner_f_bulk = [ner_f_bulk[i] for i in range(len(ner_f_bulk))
                          if i not in del_index]

        return words_bulk, ner_c_bulk, ner_f_bulk

    def __len__(self):
        return len(self.words_bulk)

    def __getitem__(self, item):
        output_tokenizer = self.tokenizer(self.words_bulk[item],
                                          is_split_into_words=True,
                                          padding="max_length",
                                          max_length=self.max_len_tokens,
                                          truncation=True)
        input_ids = output_tokenizer['input_ids']
        attention_mask_bert = output_tokenizer['attention_mask']
        word_ids = [-100 if element is None else element
                    for element in output_tokenizer.word_ids()]
        label_croase = torch.tensor(self.ner_c_bulk[item]).to(self.device)
        try:
            label_fine = torch.tensor(self.ner_f_bulk[item]).to(self.device)
        except:
            label_fine = torch.tensor(self.ner_c_bulk[item]).to(self.device)
        if self.model_type == 'detector':
            label_croase[label_croase != 0] = 1
            label_fine[label_fine != 0] = 1
        return {
            'input_ids': torch.tensor(input_ids).to(self.device),
            'words_ids': torch.tensor(word_ids).to(self.device),
            'attention_mask_bert': torch.tensor(attention_mask_bert).to(self.device),
            'label_croase': label_croase,
            'label_fine': label_fine,
            'original_token': self.words_bulk[item][:max(word_ids)+1]
        }

class TextDatasetBulkByLabel(TextDatasetBulk):
    def __init__(self, csv, window_len, step_len, device,
             tokenizer_name, max_len_tokens, goal, num_ner, model_type,
            ann_type):
        super(TextDatasetBulkByLabel, self).__init__(csv, window_len,
                                                     step_len, device,
                                                     tokenizer_name,
                                                     max_len_tokens, goal,
                                                     model_type=model_type)

        self.num_ner = num_ner
        self.ann_type = ann_type
        self.input_ids = [[] for i in range(num_ner)]
        self.words_id = [[] for i in range(num_ner)]
        self.words_id_detect = [[] for i in range(num_ner)]
        self.attention_mask_bert = [[] for i in range(num_ner)]
        self.devide_by_label()
        self.expand()

    def devide_by_label(self):
        if self.ann_type == 'croase':
            target = self.ner_c_bulk
        else:
            target = self.ner_f_bulk

        for index in range(len(target)):
            for label_index in range(self.num_ner):
                if label_index in target[index]:
                    indices = [i for i, x in enumerate(target[index])
                               if x == label_index]
                    indices_0 = [i for i, x in enumerate(target[index])
                               if x == 0]
                    indices_1 = [i for i, x in enumerate(target[index])
                                 if x != 0]
                    output_tokenizer = self.tokenizer(self.words_bulk[index],
                                                      is_split_into_words=True,
                                                      padding="max_length",
                                                      max_length=self.max_len_tokens,
                                                      truncation=True)
                    self.input_ids[label_index].append(output_tokenizer['input_ids'])
                    self.attention_mask_bert[label_index].append(output_tokenizer[
                                                                     'attention_mask'])
                    word_ids = [-100 if element not in indices else 1
                                for element in output_tokenizer.word_ids()]
                    self.words_id[label_index].append(word_ids)

                    if label_index == 0:
                        word_ids_detect = [-100 if (element not in indices_0)
                                           else 1
                                           for element in output_tokenizer.word_ids()]
                    else:
                        word_ids_detect = [-100 if (element not in indices_1)
                                           else 1
                                           for element in output_tokenizer.word_ids()]
                    self.words_id_detect[label_index].append(word_ids_detect)

    def expand(self):
        max_len = len(max(self.words_id, key=len))
        for index in range(len(self.words_id)):
            self.words_id_detect[index] = self.expand_list\
                (self.words_id_detect[index], max_len)
            self.words_id[index] = self.expand_list\
                (self.words_id[index], max_len)
            self.input_ids[index] = self.expand_list\
                (self.input_ids[index], max_len)
            self.attention_mask_bert[index] = self.expand_list\
                (self.attention_mask_bert[index], max_len)

    def expand_list(self, original_list, desired_length):
        while len(original_list) < desired_length:
            original_list += original_list[:desired_length - len(original_list)]

        return original_list

    def __len__(self):
        return len(min(self.words_id, key=len))

    def __getitem__(self, item):
        input_ids = []
        words_id = []
        words_id_detect = []
        label = []
        attention_mask_bert = []
        for index in range(self.num_ner):
            input_ids.append(self.input_ids[index][item])
            words_id.append(self.words_id[index][item])
            attention_mask_bert.append(self.attention_mask_bert[index][item])
            label.append(index)
            words_id_detect.append(self.words_id_detect[index][item])

        input_ids_tensor = torch.tensor(input_ids).to(self.device)
        words_ids_tensor = torch.tensor(words_id).to(self.device)
        words_ids_detect_tensor = torch.tensor(words_id_detect).to(self.device)
        attention_mask_bert_tensor = torch.tensor(attention_mask_bert).to(self.device)
        return {
            'input_ids': input_ids_tensor,
            'words_ids': words_ids_tensor,
            'label_croase': torch.tensor(label).view(-1).to(self.device),
            'label_fine': torch.tensor(label).view(-1).to(self.device),
            'label_detect': torch.tensor([0]+[1]*(self.num_ner-1)).view(-1).to(self.device),
            'attention_mask_bert': attention_mask_bert_tensor,
            'words_ids_detect': words_ids_detect_tensor,
            # 'label_fine': label_fine
        }

if __name__ == "__main__":
    csv = pd.read_csv('../data/train_20_en.csv')
    dataset = TextDatasetBulkByLabel(csv, window_len=30, step_len=30, device='cuda:0',
             tokenizer_name='bert-base-uncased', max_len_tokens=300, goal='train', num_ner=9, model_type='ner_tr',
                                     ann_type='croase')
    dataloader = DataLoader(dataset, batch_size=1)
    for _, data in enumerate(dataloader):
        print(data)




