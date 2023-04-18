import torch
from model_config.encoder import NerTrEncoder
from model_config.decoder import NerTrDecoder

class NerTr(torch.nn.Module):
    def __init__(self, bert_model, sim_dim,
                 num_ner, ann_type, device,
                 alignment, concatenate):
        '''
        :param bert_model: the huggingface bert model
        :param sim_dim: the dimention of the bert model like 768
        :param num_ner: the num of kinds of ner
        :param ann_type: croase or fine
        '''
        super(NerTr, self).__init__()
        self.device = device
        self.concatenate = concatenate
        self.ann_type = ann_type
        self.sim_dim = sim_dim
        self.alignment = alignment
        self.bert_model = bert_model
        self.encoder = NerTrEncoder(sim_dim=sim_dim)
        self.decoder = NerTrDecoder(num_ner=num_ner,
                                    sim_dim=sim_dim,
                                    device=self.device)
        self.normalize = torch.nn.LayerNorm(normalized_shape=sim_dim)
        self.bilstm = torch.nn.LSTM(sim_dim, sim_dim,
                                    num_layers=1,
                                    batch_first=True,
                                    bidirectional=True)
        self.linear = torch.nn.Linear(in_features=sim_dim * 2, out_features=2)
        self.activation = torch.nn.ReLU()
        self.normalize_embedding_with_prob_query = \
            torch.nn.LayerNorm(normalized_shape=sim_dim*2)

    def forward(self, data):
        '''
        :param data: the batched data from dataloader
        :return: crf loss and crf path
        '''
        if self.alignment == 'avg':
            bert_feature = self.get_bert_feature_avg(data)
        elif self.alignment == 'first':
            bert_feature = self.get_bert_feature_first(data)
        elif self.alignment == 'max':
            bert_feature = self.get_bert_feature_max(data)
        else:
            bert_feature = self.bert_model(input_ids=data['input_ids'],
                                           attention_mask=data['attention_mask_bert'])
            bert_feature = bert_feature['last_hidden_state']

        output_encoder = self.encoder(bert_feature)
        input_decoder = self.normalize(output_encoder)
        decoder_embedding, cos_sim = self.decoder(input_decoder)
        cos_sim_prob = torch.softmax(cos_sim, dim=-1)
        value, path = torch.max(cos_sim_prob, dim=-1)
        path = path.to('cpu').tolist()
        return {'path': path,
                'output': cos_sim_prob,
                }

    def prob_times_query(self, cos_sim_prob, decoder_embedding):
        b_s, prob_row, prob_clum = cos_sim_prob.shape
        _, query_row, query_clum = decoder_embedding.shape
        result = torch.zeros((b_s, prob_row, query_clum)).to(self.device)
        for b_s_index in range(b_s):
            for prob_row_index in range(prob_row):
                result_per_row = torch.zeros((prob_clum, query_clum)).to(self.device)
                for y in range(prob_clum):
                    result_per_row[y] = \
                        cos_sim_prob[b_s_index, prob_row_index, y] \
                        * decoder_embedding[b_s_index, y, :]
                # result[b_s_index, prob_row_index, :] = \
                #     torch.sum(result_per_row, dim=0, keepdim=True)
                result[b_s_index, prob_row_index, :] = \
                    torch.max(result_per_row, dim=0)[0]

        return result

    def get_bert_feature_avg(self, data):
        output_bert = self.bert_model(input_ids=data['input_ids'],
                                      attention_mask=data['attention_mask_bert'])
        last_hidden_state = output_bert['last_hidden_state']
        b_s, _, sim_dim = last_hidden_state.shape
        bert_feature_bulk = []
        for b_s_index in range(b_s):
            word_ids_one_batch = data['words_ids'][b_s_index]
            index_not_none = torch.where(word_ids_one_batch!=-100)
            last_hidden_state_one_batch = last_hidden_state[b_s_index]
            last_hidden_state_real_word = last_hidden_state_one_batch[index_not_none]
            word_ids_one_batch_not_none = word_ids_one_batch[index_not_none]
            _, indices = torch.unique(word_ids_one_batch_not_none, return_inverse=True)
            grouped_bert_embedding = []
            for i in range(torch.max(indices) + 1):
                grouped_bert_embedding.append(last_hidden_state_real_word
                                                 [indices == i])
            grouped_bert_embedding_avg = [torch.max(v, dim=0)
                                          for v in grouped_bert_embedding]
            bert_feature_bulk.append(torch.stack(grouped_bert_embedding_avg))

        return torch.stack(bert_feature_bulk)

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

    def get_bert_feature_max(self, data):
        output_bert = self.bert_model(input_ids=data['input_ids'],
                                      attention_mask=data['attention_mask_bert'])
        last_hidden_state = output_bert['last_hidden_state']
        b_s, _, sim_dim = last_hidden_state.shape
        bert_feature_bulk = []
        for b_s_index in range(b_s):
            word_ids_one_batch = data['words_ids'][b_s_index]
            index_not_none = torch.where(word_ids_one_batch != -100)
            last_hidden_state_one_batch = last_hidden_state[b_s_index]
            last_hidden_state_real_word = last_hidden_state_one_batch[index_not_none]
            word_ids_one_batch_not_none = word_ids_one_batch[index_not_none]
            _, indices = torch.unique(word_ids_one_batch_not_none, return_inverse=True)
            grouped_bert_embedding = []
            for i in range(torch.max(indices) + 1):
                grouped_bert_embedding.append(last_hidden_state_real_word
                                              [indices == i])
            grouped_bert_embedding_avg = [torch.max(v, dim=0)[0]
                                          for v in grouped_bert_embedding]
            bert_feature_bulk.append(torch.stack(grouped_bert_embedding_avg))

        return torch.stack(bert_feature_bulk)

    def get_crf_mask(self, label):
        mask = torch.where(label == 0, False, True)
        return mask

    def post_process_flow(self, ner_prob, data):
        b_s, num_tokens, num_ner = ner_prob.shape
        prob = []
        for b_s_index in range(b_s):
            word_ids_one_batch = data['words_ids'][b_s_index]
            index_not_none = torch.where(word_ids_one_batch != -100)
            ner_prob_one_batch = ner_prob[b_s_index]
            word_ids_one_batch_not_none = word_ids_one_batch[index_not_none]
            ner_prob_real_word = ner_prob_one_batch[index_not_none]
            _, indices = torch.unique(word_ids_one_batch_not_none, return_inverse=True)
            grouped_ner_prob = []
            for i in range(torch.max(indices) + 1):
                grouped_ner_prob.append(ner_prob_real_word
                                                 [indices == i])
            grouped_ner_prob_sum = [torch.sum(v, dim=0)
                                          for v in grouped_ner_prob]
            prob.append(torch.stack(grouped_ner_prob_sum))

        return torch.stack(prob)






    # def get_bert_feature(self, data):
    #     '''
    #     :param data: output of dataloader 'input_ids'[bs,1000], 'input_ids_length'[bs,618]
    #                 'attention_mask'bs.618, 'label_croase' 'label_fine'
    #     :return:
    #     '''
    #     output_bert = self.bert_model(input_ids=data['input_ids'],
    #                                   attention_mask=data['attention_mask_bert'])
    #     last_hidden_state = output_bert['last_hidden_state']
    #     b_s, true_length, _ = last_hidden_state.shape
    #     bert_avg = []
    #     for b_s_index in range(b_s):
    #         last_hidden_state_one_batch = last_hidden_state[b_s_index]
    #         last_hidden_state_real = last_hidden_state_one_batch[
    #                                  :len(data['input_ids_length'][b_s_index])]
    #         temp = []
    #         i = 0
    #         for length in data['input_ids_length'][b_s_index][
    #             data['input_ids_length'][b_s_index].nonzero().squeeze()]:
    #             temp.append(torch.mean(last_hidden_state_real[i:i + length], dim=0))
    #             i += length
    #         stack = torch.stack(temp, dim=0)
    #         padding = (0, 0, 0, self.max_len_words - len(stack))
    #         bert_avg.append(torch.nn.functional.pad(stack, padding))
    #
    #     return torch.stack(bert_avg, dim=0)
