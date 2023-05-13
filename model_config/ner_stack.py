import torch
from model_config.encoder import NerTrEncoder
from model_config.decoder import NerTrDecoder
from model_config.ner_tr import NerTr
from torchcrf import CRF
# from fastNLP.modules import ConditionalRandomField

class NerStack(NerTr):
    def __init__(self, bert_model, sim_dim,
                 num_ner, ann_type, device,
                 alignment, concatenate,
                 num_encoder):
        super(NerStack, self).__init__(bert_model, sim_dim,
                 num_ner, ann_type, device,
                 alignment, concatenate,
                 num_encoder, encoder_rela=False,
                 encoder_bert=True)
        self.linear = torch.nn.Linear(in_features=sim_dim * num_encoder,
                                      out_features=num_ner)
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

        try:
            output_encoder = [encoder(bert_feature) for encoder in self.encoder_bert]
        except:
            output_encoder = [encoder(bert_feature.float())
                              for encoder in self.encoder_bert]
        input_linear = torch.cat(output_encoder, dim=-1)
        output_linear = self.linear(input_linear)
        input_crf = torch.softmax(output_linear, dim=-1)
        output_crf = self.crf(input_crf,
                              data['label_'+self.ann_type].view(input_crf.shape[:-1]),
                              reduction='mean')
        path = self.crf.decode(input_crf)
        return {'loss': -1 * output_crf,
                'path': path}














