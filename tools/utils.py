import torch
from torch.utils.data.dataloader import DataLoader
from model_components.dataset_hipe import TextDataset
import pandas as pd
from transformers import BertModel, BertConfig
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

if __name__ == "__main__":
    config = BertConfig.from_pretrained('camembert-base')
    config.max_position_embeddings = 1000
    bert = BertModel.from_pretrained('camembert-base', config=config)
    csv = pd.read_csv('../data/test_fr.csv')
    dataset = TextDataset(csv, max_len_tokens=618)
    dataloader = DataLoader(dataset, batch_size=4)
    for data in tqdm(dataloader):
        data_preprocess(data, bert=bert)