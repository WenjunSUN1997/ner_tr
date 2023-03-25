import torch
from model_config.encoder import NerTrEncoder
from model_config.decoder import NerTrDecoder
from TorchCRF import CRF
from fastNLP.modules import ConditionalRandomField

class NerTr(torch.nn.Module):
    def __init__(self, bert_model, sim_dim, num_ner, ann_type, device, alignment,
                 concatenate):
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
        self.decoder = NerTrDecoder(num_ner=num_ner, sim_dim=sim_dim, device=self.device)
        self.linear_add = torch.nn.Linear(in_features=sim_dim, out_features=num_ner)
        self.linear_con = torch.nn.Linear(in_features=sim_dim * 2, out_features=num_ner)
        self.normalize = torch.nn.LayerNorm(normalized_shape=sim_dim)
        self.crf = CRF(num_ner, batch_first=True)
        # self.crf = ConditionalRandomField(num_tags=num_ner,
        #                                   include_start_end_trans=True)
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
        #output_encoder: [b_s, max_num_token, sim_dim]
        output_encoder = self.encoder(bert_feature)
        #decoder_embedding:[b_s, num_kind_of_ner, sim_dim], the embedding of quert
        #cos_sim: [b_s, max_num_token, num_kind_of_ner],
        #the cos_sim between output_encoder and decoder_embedding, e.g. the similarity of
        #tokens and query embedding
        decoder_embedding, cos_sim = self.decoder(output_encoder)
        #transform cos_sim into prob
        cos_sim_prob = torch.softmax(cos_sim, dim=-1)
        prob_query = self.prob_times_query(cos_sim, decoder_embedding)
        # prob_query = self.prob_times_query(cos_sim_prob, decoder_embedding)
        if self.concatenate == 'add':
            embedding_with_prob_query = output_encoder + prob_query
            embedding_with_prob_query = self.normalize(embedding_with_prob_query)
            embedding_with_prob_query = self.activation(embedding_with_prob_query)
            ner_prob = self.linear_add(embedding_with_prob_query)
        elif self.concatenate == 'con':
            embedding_with_prob_query = torch.cat((output_encoder, prob_query), dim=2)
            embedding_with_prob_query = self.normalize_embedding_with_prob_query(
                embedding_with_prob_query)
            embedding_with_prob_query = self.activation(embedding_with_prob_query)
            ner_prob = self.linear_con(embedding_with_prob_query)

        ner_prob = torch.log_softmax(ner_prob, dim=-1)
        if self.alignment == 'flow':
            ner_prob = self.post_process_flow(cos_sim_prob, data)

        # mask = torch.ones(data['label_'+self.ann_type].shape, dtype=torch.bool)
        # crf_loss = self.crf(ner_prob, data['label_'+self.ann_type],
        #                     mask=mask.to(self.device))
        # crf_loss = torch.sum(crf_loss, dim=0)
        # path = self.crf.viterbi_decode(ner_prob, mask=mask.to(self.device))[0]
        # return {'loss': crf_loss,
        #         'path': path}
        crf_loss = self.crf(ner_prob, data['label_' + self.ann_type], reduction='mean')
        path = self.crf.decode(ner_prob)
        return {'loss': -1 * crf_loss,
                'path': path}

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
            grouped_bert_embedding_avg = [torch.mean(v, dim=0)
                                          for v in grouped_bert_embedding]
            bert_feature_bulk.append(torch.stack(grouped_bert_embedding_avg))

        return torch.stack(bert_feature_bulk)

    def get_bert_feature_first(self, data):
        input_ids = data['input_ids']
        output_bert = self.bert_model(input_ids=data['input_ids'],
                                      attention_mask=data['attention_mask_bert'])['last_hidden_state']
        b_s, token_num, sim_dim = output_bert.shape
        result = []
        for b_s_index in range(b_s):
            word_ids_one_batch = data['words_ids'][b_s_index]
            index_not_none = torch.where(word_ids_one_batch != -100)
            word_ids_one_batch_not_none = word_ids_one_batch[index_not_none]
            input_ids_one_batch_real = input_ids[b_s_index][index_not_none]
            _, indices = torch.unique(word_ids_one_batch_not_none, return_inverse=True)
            grouped_token_id_first = []
            for i in range(torch.max(indices) + 1):
                grouped_token_id_first.append(input_ids_one_batch_real
                                                 [indices == i])
            grouped_bert_embedding_first = [v[0] for v in grouped_token_id_first]
            result.append(torch.stack(grouped_bert_embedding_first))

        bert_embedding = self.bert_model(input_ids=torch.stack(result))['last_hidden_state']
        return bert_embedding

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






