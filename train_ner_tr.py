from model_components.dataloader_ner_tr import get_dataloader
from model_config.ner_tr import NerTr
from transformers import BertModel, BertConfig
import argparse
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from model_components.validator_ner_tr import validate

def train(lang, window_len, step_len, max_len_tokens, tokenizer_name, index_out,
          bert_model_name, num_ner, ann_type, sim_dim, device, batch_size, alignment,
          concatenate):
    LR = 1e-5
    epoch = 10000

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
    config = BertConfig.from_pretrained(bert_model_name)
    config.max_position_embeddings = max_len_tokens
    bert_model = BertModel.from_pretrained(bert_model_name, config=config)
    ner_model = NerTr(bert_model=bert_model, sim_dim=sim_dim,
                      num_ner=num_ner, ann_type=ann_type, device=device,
                      alignment=alignment, concatenate=concatenate)
    ner_model.to(device)
    ner_model.train()
    optimizer = torch.optim.Adam(params=ner_model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                  factor=0.5, patience=1, verbose=True)
    for epoch_num in range(epoch):
        loss_all = []
        for data in tqdm(dataloader_train):
            output = ner_model(data)
            loss = output['loss']
            print(loss)
            # print(data['label_croase'])
            # print(output['path'])
            loss_all.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_epoch = sum(loss_all) / len(loss_all)
        print(loss_epoch)
        loss_val = validate(model=ner_model,
                            dataloader=dataloader_test, num_ner=num_ner,
                            ann_type=ann_type, index_out=index_out)
        scheduler.step(loss_val)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default='fre')
    parser.add_argument("--window_len", default=10)
    parser.add_argument("--step_len", default=5)
    parser.add_argument("--max_len_tokens", default=200)
    parser.add_argument("--tokenizer_name", default='camembert-base')
    parser.add_argument("--bert_model_name", default='camembert-base')
    parser.add_argument("--num_ner", default=9)
    parser.add_argument("--alignment", default='flow', choices=['avg', 'flow',
                                                                'max', 'first'])
    parser.add_argument("--concatenate", default='flow', choices=['add', 'con'])
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