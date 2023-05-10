from model_components.dataloader_ner_tr import get_dataloader
from model_config.ner_tr import NerTr
from model_config.ner_detr import NerDetr
from model_config.ner_detector import NerDetector
from transformers import BertModel, BertConfig
import argparse
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from model_components.validator_ner_tr import validate

torch.manual_seed(3407)

def train(lang, window_len, step_len, max_len_tokens, tokenizer_name, index_out,
          bert_model_name, num_ner, ann_type, sim_dim, device, batch_size, alignment,
          concatenate, model_type, weight_o, train_bert, data_aug, num_encoder):
    LR = 2e-5
    epoch = 10000

    dataloader_train = get_dataloader(lang=lang, goal='train',
                                      window_len=window_len,
                                      step_len=step_len,
                                      max_len_tokens=max_len_tokens,
                                      tokenizer_name=tokenizer_name,
                                      batch_size=batch_size,
                                      device=device,
                                      num_ner=num_ner,
                                      model_type=model_type,
                                      data_aug=data_aug)
    dataloader_dev = get_dataloader(lang=lang, goal='dev',
                                    window_len=window_len,
                                    step_len=window_len,
                                    max_len_tokens=max_len_tokens,
                                    tokenizer_name=tokenizer_name,
                                    batch_size=batch_size,
                                    device=device,
                                    num_ner=num_ner,
                                    model_type=model_type,
                                    data_aug=data_aug)
    dataloader_test = get_dataloader(lang=lang, goal='test',
                                     window_len=window_len,
                                     step_len=window_len,
                                     max_len_tokens=max_len_tokens,
                                     tokenizer_name=tokenizer_name,
                                     batch_size=batch_size,
                                     device=device,
                                     num_ner=num_ner,
                                     model_type=model_type,
                                     data_aug=data_aug)

    bert_model = BertModel.from_pretrained(bert_model_name)
    if model_type == 'ner_tr':
        ner_model = NerTr(bert_model=bert_model,
                          sim_dim=sim_dim,
                          num_ner=num_ner,
                          ann_type=ann_type,
                          device=device,
                          alignment=alignment,
                          concatenate=concatenate,
                          num_encoder=num_encoder)
    elif model_type == 'detector':
        ner_model = NerDetector(bert_model=bert_model,
                                sim_dim=sim_dim,
                                num_ner=num_ner)

    for param in ner_model.bert_model.parameters():
        param.requires_grad = train_bert

    ner_model.to(device)
    ner_model.train()
    optimizer = torch.optim.AdamW(ner_model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer,
                                  mode='min',
                                  factor=0.5,
                                  patience=2,
                                  verbose=True)
    weight_ner = torch.ones(num_ner)
    weight_ner[0] = weight_o
    loss_func = torch.nn.CrossEntropyLoss(weight=weight_ner.to(device))
    for epoch_num in range(epoch):
        print(epoch_num)
        loss_all = []
        for step, data in tqdm(enumerate(dataloader_train), total=len(dataloader_train)):
            # break
            output = ner_model(data)
            loss = loss_func(output['output'].view(-1, num_ner),
                             data['label_' + ann_type].view(-1))
            loss_all.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (step+1) % 1000 == 0:
                print(epoch_num)
                print('dev:')
                output_dev = validate(model=ner_model,
                                    dataloader=dataloader_dev,
                                    num_ner=num_ner,
                                    ann_type=ann_type,
                                    index_out=index_out,
                                    lang=lang,
                                    epoch_num=epoch_num,
                                    model_type=model_type,
                                    loss_func=loss_func)
                scheduler.step(output_dev['loss_ce'])

        loss_epoch = sum(loss_all) / len(loss_all)
        print(loss_epoch)
        print('val:')
        output_val = validate(model=ner_model,
                              dataloader=dataloader_test,
                              num_ner=num_ner,
                              ann_type=ann_type,
                              index_out=index_out,
                              lang=lang,
                              epoch_num=epoch_num,
                              model_type=model_type,
                              loss_func=loss_func)
        scheduler.step(output_val['loss_ce'])
        print('val loss', output_val['loss_ce'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default='fre', choices=['fre', 'wnut', 'conll',
                                                          'conll_au', 'phoner'])
    parser.add_argument("--window_len", default=100)
    parser.add_argument("--step_len", default=100)
    parser.add_argument("--max_len_tokens", default=400)
    default_bert = 'dbmdz/bert-base-historic-multilingual-64k-td-cased'
    parser.add_argument("--bert_model_name", default=default_bert)
    parser.add_argument("--num_ner", default=9)
    parser.add_argument("--alignment", default='first', choices=['avg', 'flow',
                                                                 'max', 'first'])
    parser.add_argument("--concatenate", default='con', choices=['add', 'con'])
    parser.add_argument("--ann_type", default='croase')
    parser.add_argument("--sim_dim", default=768)
    parser.add_argument("--device", default='cuda:0')
    parser.add_argument("--batch_size", default=2)
    parser.add_argument("--index_out", default=0)
    parser.add_argument("--data_aug", default=0)
    parser.add_argument("--num_encoder", default=2)
    parser.add_argument("--weight_o", default=10.0)
    parser.add_argument("--train_bert", default='0')
    parser.add_argument("--model_type", default='ner_tr', choices=['ner_tr', 'detector'])
    args = parser.parse_args()
    print(args)
    alignment = args.alignment
    lang = args.lang
    weight_o = float(args.weight_o)
    concatenate = args.concatenate
    train_bert = True if args.train_bert == '1' else False
    data_aug = True if args.data_aug == '1' else False
    index_out = int(args.index_out)
    batch_size = int(args.batch_size)
    window_len = int(args.window_len)
    step_len = int(args.step_len)
    max_len_tokens = int(args.max_len_tokens)
    tokenizer_name = args.bert_model_name
    bert_model_name = args.bert_model_name
    num_ner = int(args.num_ner)
    ann_type = args.ann_type
    sim_dim = int(args.sim_dim)
    num_encoder = int(args.num_encoder)
    model_type = args.model_type
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
          concatenate=concatenate,
          model_type=model_type,
          weight_o=weight_o,
          train_bert=train_bert,
          data_aug=data_aug,
          num_encoder=num_encoder)
