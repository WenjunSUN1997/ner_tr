from model_components.dataloader_ner_tr import get_dataloader
from model_config.ner_tr import NerTr
from model_config.ner_detr import NerDetr
from transformers import BertModel
import argparse
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from model_components.validator_ner_detr import validate
from model_components.loss_func import HungaryLoss

def train(lang, window_len, step_len, max_len_tokens, tokenizer_name, index_out,
          bert_model_name, num_ner, ann_type, sim_dim, device, batch_size, alignment,
          concatenate):
    LR = 1e-5
    epoch = 10000
    loss_func = HungaryLoss(num_ner=num_ner, device=device, window_len=window_len)

    dataloader_train = get_dataloader(lang=lang, goal='train',
                                      window_len=window_len,
                                      step_len=step_len,
                                      max_len_tokens=max_len_tokens,
                                      tokenizer_name=tokenizer_name,
                                      batch_size=batch_size,
                                      device=device)
    dataloader_test = get_dataloader(lang=lang, goal='test',
                                     window_len=window_len,
                                     step_len=window_len,
                                      max_len_tokens=max_len_tokens,
                                      tokenizer_name=tokenizer_name,
                                     batch_size=batch_size,
                                     device=device)
    bert_model = BertModel.from_pretrained(bert_model_name)
    ner_model = NerDetr(bert_model=bert_model, sim_dim=sim_dim,
                      num_ner=num_ner, ann_type=ann_type, device=device,
                      alignment=alignment, concatenate=concatenate, win_len=window_len)
    for param in ner_model.bert_model.parameters():
        param.requires_grad = True

    ner_model.to(device)
    ner_model.train()
    optimizer = torch.optim.AdamW(params=ner_model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                  factor=0.5, patience=1, verbose=True)
    for epoch_num in range(epoch):
        loss_all = []
        for step, data in tqdm(enumerate(dataloader_train), total=len(dataloader_train)):
            output = ner_model(data)
            loss = loss_func(output, data['label_' + ann_type])
            print(loss)
            loss_all.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print()

        loss_epoch = sum(loss_all) / len(loss_all)
        print(loss_epoch)
        print('val:')
        loss_val = validate(model=ner_model,
                            dataloader=dataloader_test, num_ner=num_ner,
                            ann_type=ann_type, index_out=index_out)
        scheduler.step(loss_val)
        print('val loss', loss_val)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default='fre')
    parser.add_argument("--window_len", default=10)
    parser.add_argument("--step_len", default=10)
    parser.add_argument("--max_len_tokens", default=40)
    parser.add_argument("--tokenizer_name", default='camembert-base')
    parser.add_argument("--bert_model_name", default='camembert-base')
    parser.add_argument("--num_ner", default=9)
    parser.add_argument("--alignment", default='first', choices=['avg', 'flow',
                                                                'max', 'first'])
    parser.add_argument("--concatenate", default='add', choices=['add', 'con'])
    parser.add_argument("--ann_type", default='croase')
    parser.add_argument("--sim_dim", default=768)
    parser.add_argument("--device", default='cuda:0')
    parser.add_argument("--batch_size", default=8)
    parser.add_argument("--index_out", default=0)
    args = parser.parse_args()
    print(args)
    alignment = args.alignment
    lang = args.lang
    concatenate = args.concatenate
    index_out = int(args.index_out)
    batch_size = int(args.batch_size)
    window_len = int(args.window_len)
    step_len = int(args.step_len)
    max_len_tokens = int(args.max_len_tokens)
    tokenizer_name = args.tokenizer_name
    bert_model_name =args.bert_model_name
    num_ner = int(args.num_ner)
    ann_type = args.ann_type
    sim_dim = int(args.sim_dim)
    device = args.device
    train(lang=lang,
          window_len=window_len,
          step_len=step_len,
          max_len_tokens=max_len_tokens,
          tokenizer_name=tokenizer_name,
          index_out=index_out,
          bert_model_name=bert_model_name,
          num_ner=num_ner,
          ann_type=ann_type,
          sim_dim=sim_dim,
          device=device,
          batch_size=batch_size,
          alignment=alignment,
          concatenate=concatenate)