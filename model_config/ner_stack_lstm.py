import torch
from model_config.encoder import NerTrEncoder
from model_config.decoder import NerTrDecoder
from model_config.ner_tr import NerTr
# from fastNLP.modules import ConditionalRandomField

class NerStackLSTM(NerTr):
    def __init__(self, bert_model, sim_dim, num_ner, ann_type, device, alignment,
                 concatenate, win_len, num_ner_kind=9):
        '''

        :param bert_model: the huggingface bert model
        :param sim_dim: the dimention of the bert model like 768
        :param num_ner: the num of kinds of ner
        :param ann_type: croase or fine
        '''
        super(NerStackLSTM, self).__init__(bert_model, sim_dim, num_ner, ann_type, device, alignment,
                 concatenate)
        self.win_len = win_len
        self.bilstm_tok = torch.nn.LSTM(sim_dim, sim_dim*2, num_layers=2, batch_first=True,
                                    bidirectional=True)
        self.bilstm_ner_type = torch.nn.ModuleList()
        for bilstm_index in range(num_ner_kind):
            self.bilstm_ner_type.append(torch.nn.LSTM(sim_dim, sim_dim*2, num_layers=2,
                                                      batch_first=True,
                                                      bidirectional=True))
        self.activation = torch.nn.Tanh()

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
        bert_feature = self.dropout(bert_feature)
        bert_feature = self.normalize(bert_feature)










