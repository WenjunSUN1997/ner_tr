import torch
from model_config.ner_tr import NerTr
from torchcrf import CRF

class NerCrf(NerTr):
    def __init__(self, bert_model, sim_dim, num_ner, ann_type, device, alignment,
                 concatenate):
        super(NerCrf, self).__init__(bert_model, sim_dim,
                                     num_ner, ann_type,
                                     device, alignment,
                                     concatenate)
        self.crf = CRF(num_ner, batch_first=True)

    def forward(self, data):
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
        input_decoder = self.normalize(bert_feature + output_encoder)
        decoder_embedding, cos_sim = self.decoder(input_decoder)
        cos_sim_prob = torch.softmax(cos_sim, dim=-1)
        crf_loss = self.crf(cos_sim_prob,
                            data['label_'+self.ann_type].view(cos_sim_prob.shape[:-1]),
                            reduction='mean')
        path = self.crf.decode(cos_sim_prob)
        return {'loss': -1 * crf_loss,
                'path': path,
                'output': cos_sim_prob}





