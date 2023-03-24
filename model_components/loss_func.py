import torch
from TorchCRF import CRF

class NerLoss(torch.nn.Module):
    def __init__(self, num_ner):
        super(NerLoss, self).__init__()
        self.crf = CRF(num_ner)
        self.crf_loss = self.crf.loss_function

    def forward(self, ner_prob, label):
        pass
