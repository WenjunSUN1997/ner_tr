import torch
from transformers import AutoTokenizer, BertModel, BertConfig

class NerTrDecoder(torch.nn.Module):
    def __init__(self, num_ner, sim_dim, device, bert_name='bert-base-uncased'):
        super(NerTrDecoder, self).__init__()
        self.num_ner = num_ner
        self.device = device
        self.obj_query_embedding = torch.nn.Embedding(num_ner, sim_dim)
        self.decoder_layer = torch.nn.TransformerDecoderLayer(d_model=sim_dim,
                                                              nhead=8,
                                                              batch_first=True)
        self.decoder = torch.nn.TransformerDecoder(self.decoder_layer, num_layers=2)
        self.normalize = torch.nn.LayerNorm(normalized_shape=sim_dim)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_name)
        # self.bert_model = BertModel.from_pretrained(bert_name)

    def get_obj_query_embedding(self, batch_size):
        obj_query_index = torch.tensor([x for x in range(self.num_ner)])
        obj_query_embedding = self.obj_query_embedding(obj_query_index.to(self.device))
        obj_query_embedding_batched = obj_query_embedding.repeat(batch_size, 1, 1)
        return obj_query_embedding_batched.to(self.device)

    def forward_query_ner(self, semantic):
        obj_query_embedding_batched = self.get_obj_query_embedding(semantic.shape[0])
        query_result = self.decoder(memory=semantic,
                                    tgt=obj_query_embedding_batched)
        return query_result

    def get_cos_sim(self, query_result, text_embedding):
        cos_sim = torch.cosine_similarity(query_result.unsqueeze(1),
                                          text_embedding.unsqueeze(-2),
                                          dim=-1)
        return cos_sim

    def forward(self, semantic):
        query_result = self.forward_query_ner(semantic)
        # query_result = self.forward_query_label(semantic)
        # query_result = self.get_labels_query_embedding(semantic.shape[0])
        cos_sim = self.get_cos_sim(query_result, semantic)
        return query_result, cos_sim

    # def get_query_label(self):
    #     labels_text = ['pas de entité',
    #                 'début de individu ou groupe de personnes', 'milieu de individu ou groupe de personnes',
    #               'début de commercial, éducatif, divertissement, gouvernement, médias, médical-science, non gouvernemental, religieux, sports',
    #             'milieu de commercial, éducatif, divertissement, gouvernement, médias, médical-science, non gouvernemental, religieux, sports',
    #               'début de adresse, territoire ayant une frontière géopolitique tel que ville, pays, région, continent, nation, état ou province',
    #                    'milieu de adresse, territoire ayant une frontière géopolitique tel que ville, pays, région, continent, nation, état ou province',
    #               'début de les produits médiatiques tels que les journaux, les magazines, les émissions, etc.',
    #                    'milieu de les produits médiatiques tels que les journaux, les magazines, les émissions, etc.']
    #     labels_text_en = ['no entity', 'begin person', 'inside person',
    #                    'begin organisation', 'inside organisation',
    #                    'begin location', 'inside location',
    #                    'begin miscellaneous', 'inside miscellaneous']
    #
    #     output_tokenizer = self.tokenizer(labels_text_en,
    #                                       truncation=True,
    #                                       padding="max_length",
    #                                       max_length=6,
    #                                       return_tensors='pt')
    #     output_bert = self.bert_model(input_ids=output_tokenizer['input_ids']
    #                                   .to(self.device),
    #                                   attention_mask=output_tokenizer['attention_mask']
    #                                   .to(self.device))
    #     labels_semantic = output_bert['last_hidden_state'][:, 0, :]
    #     return labels_semantic
    #
    # def get_labels_query_embedding(self, batch_size):
    #     obj_query_embedding = self.get_query_label()
    #     obj_query_embedding_batched = obj_query_embedding.repeat(batch_size, 1, 1)
    #     return obj_query_embedding_batched.to(self.device)
    #
    # def forward_query_label(self, semantic):
    #     obj_query_embedding_batched = self.get_labels_query_embedding(semantic.shape[0])
    #     query_result = self.decoder(memory=semantic,
    #                                 tgt=obj_query_embedding_batched)
    #     return query_result
