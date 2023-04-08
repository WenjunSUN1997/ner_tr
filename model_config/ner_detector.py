import torch

class NerDetector(torch.nn.Module):
    def __init__(self, bert_model, sim_dim, num_ner):
        super(NerDetector, self).__init__()
        self.bert_model = bert_model
        self.bilstm = torch.nn.LSTM(sim_dim, sim_dim,
                                            num_layers=1,
                                            batch_first=True,
                                            bidirectional=True)
        self.linear = torch.nn.Linear(in_features=sim_dim*2, out_features=num_ner)

    def forward(self, data):
        bert_feature = self.get_bert_feature_first(data)
        output_bilstm = self.bilstm(bert_feature)[0]
        prob = self.linear(output_bilstm)
        prob = torch.softmax(prob, dim=-1)
        path = torch.argmax(prob, dim=-1).to('cpu').tolist()
        return {'path': path,
                'output': prob}

    def get_bert_feature_first(self, data):
        input_ids = data['input_ids']
        if len(data['input_ids'].shape) == 3:
            input_ids = data['input_ids'].view(-1, input_ids.shape[-1])
            attention_mask = data['attention_mask_bert'].view(-1, input_ids.shape[-1])
            words_ids = data['words_ids'].view(-1, input_ids.shape[-1])
        else:
            input_ids = data['input_ids']
            attention_mask = data['attention_mask_bert']
            words_ids = data['words_ids']

        output_bert = self.bert_model(input_ids=input_ids,
                                      attention_mask=attention_mask
                                      )['last_hidden_state']
        b_s, token_num, sim_dim = output_bert.shape
        result = []
        for b_s_index in range(b_s):
            word_ids_one_batch = words_ids[b_s_index]
            index_not_none = torch.where(word_ids_one_batch != -100)
            word_ids_one_batch_not_none = word_ids_one_batch[index_not_none]
            bert_one_batch_real = output_bert[b_s_index][index_not_none]
            _, indices = torch.unique(word_ids_one_batch_not_none, return_inverse=True)
            grouped_token_id_first = []
            for i in range(torch.max(indices) + 1):
                grouped_token_id_first.append(bert_one_batch_real
                                                 [indices == i])
            if len(data['input_ids'].shape) != 3:
                grouped_bert_embedding_first = [v[0] for v in grouped_token_id_first]
            else:
                grouped_bert_embedding_first = [torch.mean(grouped_token_id_first[0],
                                                           dim=0)]
                # grouped_bert_embedding_first = [v[0] for v in grouped_token_id_first]

            result.append(torch.stack(grouped_bert_embedding_first))

        # bert_embedding = self.bert_model(input_ids=torch.stack(result))['last_hidden_state']
        if len(data['input_ids'].shape) != 3:
            return torch.stack(result)
        else:
            return torch.stack(result).squeeze(1).unsqueeze(0)