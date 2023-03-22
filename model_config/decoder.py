import torch

class NerTrDecoder(torch.nn.Module):
    def __init__(self, num_ner, sim_dim, device):
        super(NerTrDecoder, self).__init__()
        self.num_ner = num_ner
        self.device = device
        self.obj_query_embedding = torch.nn.Embedding(num_ner, sim_dim)
        self.decoder_layer = torch.nn.TransformerDecoderLayer(d_model=sim_dim,
                                                              nhead=8,
                                                              batch_first=True)
        self.decoder = torch.nn.TransformerDecoder(self.decoder_layer, num_layers=2)
        self.normalize = torch.nn.LayerNorm(normalized_shape=sim_dim)

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
        cos_sim = self.get_cos_sim(query_result, semantic)
        return query_result, cos_sim
