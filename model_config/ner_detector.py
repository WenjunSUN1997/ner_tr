import torch
from model_config.encoder import NerTrEncoder
from model_config.decoder import NerTrDecoder
from torchcrf import CRF

class NerDetector(torch.nn.Module):
    def __init__(self, bert_model, sim_dim, num_ner):
        super(NerDetector, self).__init__()
        self.bert_model = bert_model
        self.encoder = NerTrEncoder(sim_dim=sim_dim)
        self.decoder = NerTrDecoder(num_ner=2,
                                    sim_dim=sim_dim,
                                    device='cuda:0')
        self.normalize = torch.nn.LayerNorm(normalized_shape=sim_dim)
        self.linear = torch.nn.Linear(in_features=sim_dim, out_features=2)

    def forward(self, data):
        bert_feature = self.get_bert_feature_first(data)
        output_encoder = self.encoder(bert_feature.squeeze(0))
        input_linear = self.normalize(output_encoder)
        output_linear = self.linear(input_linear)
        ner_prob = torch.softmax(output_linear, dim=-1)
        path = torch.argmax(ner_prob, dim=-1).to('cpu')
        path = path.tolist()
        return {'path': path,
                'ner_prob': ner_prob,
                'loss': 1}

    def get_bert_feature_first(self, data):
        try:
            input_ids = data['input_ids_detect']
            words_ids = data['words_ids_detect']
            attention_mask = data['attention_mask_bert_detect']
        except:
            input_ids = data['input_ids']
            words_ids = data['words_ids']
            attention_mask = data['attention_mask_bert']

        if len(input_ids.shape) == 3:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            attention_mask = attention_mask.view(-1, input_ids.shape[-1])
            words_ids = words_ids.view(-1, input_ids.shape[-1])

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

            grouped_bert_embedding_first = [v[0] for v in grouped_token_id_first]

            result.append(torch.stack(grouped_bert_embedding_first))

        # bert_embedding = self.bert_model(input_ids=torch.stack(result))['last_hidden_state']
        if len(data['input_ids'].shape) != 3:
            return torch.stack(result)
        else:
            return torch.stack(result).squeeze(1).unsqueeze(0)