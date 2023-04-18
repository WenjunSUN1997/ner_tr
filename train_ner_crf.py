from model_components.dataloader_ner_tr import get_dataloader
from model_config.ner_tr import NerTr
from model_config.ner_crf import NerCrf
from model_config.ner_detr import NerDetr
from transformers import BertModel, BertConfig
import argparse
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from model_components.validator_ner_tr import validate

def train(lang, window_len, step_len, max_len_tokens, tokenizer_name, index_out,
          bert_model_name, num_ner, ann_type, sim_dim, device, batch_size, alignment,
          concatenate):
    LR = 2e-3
    epoch = 10000

    dataloader_train = get_dataloader(lang=lang, goal='train_crf',
                                      window_len=window_len,
                                      step_len=step_len,
                                      max_len_tokens=max_len_tokens,
                                      tokenizer_name=tokenizer_name,
                                      batch_size=batch_size,
                                      device=device,
                                      num_ner=num_ner)
    dataloader_dev = get_dataloader(lang=lang, goal='dev',
                                      window_len=window_len,
                                      step_len=window_len,
                                      max_len_tokens=max_len_tokens,
                                      tokenizer_name=tokenizer_name,
                                      batch_size=batch_size,
                                      device=device,
                                    num_ner=num_ner)
    dataloader_test = get_dataloader(lang=lang, goal='test',
                                     window_len=window_len,
                                     step_len=window_len,
                                      max_len_tokens=max_len_tokens,
                                      tokenizer_name=tokenizer_name,
                                     batch_size=batch_size,
                                     device=device,
                                     num_ner=num_ner)
    config = BertConfig.from_pretrained(bert_model_name)
    bert_model = BertModel(config=config)
    ner_crf_model = NerCrf(bert_model=bert_model,
                           sim_dim=sim_dim,
                           num_ner=num_ner,
                           ann_type=ann_type,
                           device=device,
                           alignment=alignment,
                           concatenate=concatenate)
    # pretrained_dict = torch.load('model_zoo/conll_111_78.07.pth')
    # ner_crf_model.load_state_dict(pretrained_dict, strict=False)
    for name, param in ner_crf_model.named_parameters():
        if 'crf' not in name:
            param.requires_grad = False

    ner_crf_model.to(device)
    ner_crf_model.train()
    optimizer = torch.optim.AdamW(ner_crf_model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer,
                                  mode='min',
                                  factor=0.3,
                                  patience=3,
                                  verbose=True)
    for epoch_num in range(epoch):
        print(epoch_num)
        loss_all = []
        for step, data in tqdm(enumerate(dataloader_train), total=len(dataloader_train)):
            # break
            output = ner_crf_model(data)
            loss = output['loss']
            print(loss)
            loss_all.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (step+1) % 4000 == 0:
                print(epoch_num)
                print('dev:')
                loss_dev = validate(model=ner_crf_model,
                                    dataloader=dataloader_dev,
                                    num_ner=num_ner,
                                    ann_type=ann_type,
                                    index_out=index_out,
                                    lang=lang,
                                    epoch_num=epoch_num)

        loss_epoch = sum(loss_all) / len(loss_all)
        print(loss_epoch)
        print('val:')
        output_val = validate(model=ner_crf_model,
                              dataloader=dataloader_test,
                              num_ner=num_ner,
                              ann_type=ann_type,
                              index_out=index_out,
                              lang=lang,
                              epoch_num=epoch_num)
        scheduler.step(output_val['loss'])
        print('val loss', output_val['loss'])
        print('\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default='conll', choices=['fre', 'wnut', 'conll',
                                                               'conll_au'])
    parser.add_argument("--window_len", default=30)
    parser.add_argument("--step_len", default=30)
    parser.add_argument("--max_len_tokens", default=150)
    parser.add_argument("--tokenizer_name", default='bert-base-cased')
    parser.add_argument("--bert_model_name", default='bert-base-cased')
    parser.add_argument("--num_ner", default=9)
    parser.add_argument("--alignment", default='first', choices=['avg', 'flow',
                                                                'max', 'first'])
    parser.add_argument("--concatenate", default='con', choices=['add', 'con'])
    parser.add_argument("--ann_type", default='croase')
    parser.add_argument("--sim_dim", default=768)
    parser.add_argument("--device", default='cuda:0')
    parser.add_argument("--batch_size", default=2)
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