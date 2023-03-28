import torch
from model_config.encoder import NerTrEncoder
from model_config.decoder import NerTrDecoder
from model_config.ner_tr import NerTr
# from fastNLP.modules import ConditionalRandomField

class NerDetr(NerTr):
    def __init__(self, bert_model, sim_dim, num_ner, ann_type, device, alignment,
                 concatenate, win_len, num_ner_kind=5):
        '''
        :param bert_model: the huggingface bert model
        :param sim_dim: the dimention of the bert model like 768
        :param num_ner: the num of kinds of ner
        :param ann_type: croase or fine
        '''
        super(NerDetr, self).__init__(bert_model, sim_dim, num_ner, ann_type, device, alignment,
                 concatenate)
        self.win_len = win_len
        self.linear_class = torch.nn.Linear(in_features=sim_dim, out_features=num_ner_kind)
        self.linear_pos = torch.nn.Linear(in_features=sim_dim, out_features=2)
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
        bert_feature = self.normalize(bert_feature )
        output_encoder = self.encoder(bert_feature)
        #decoder_embedding:[b_s, num_kind_of_ner, sim_dim], the embedding of quert
        #cos_sim: [b_s, max_num_token, num_kind_of_ner],
        #the cos_sim between output_encoder and decoder_embedding, e.g. the similarity of
        #tokens and query embedding
        decoder_embedding, cos_sim = self.decoder(
            self.normalize(bert_feature+output_encoder))
        decoder_embedding = self.dropout(decoder_embedding)
        decoder_embedding = self.activation(decoder_embedding)
        class_pre = self.linear_class(decoder_embedding)
        class_pre = torch.softmax(class_pre, dim=2)
        pos_pre = self.linear_pos(decoder_embedding)
        pos_pre = self.win_len * torch.sigmoid(pos_pre)
        return {'class_pre': class_pre, 'pos_pre': pos_pre}








