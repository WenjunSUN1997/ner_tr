from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
import pandas as pd

def validate(dataloader, model, ann_type, index_out,
             num_ner, lang, epoch_num, model_type):
    labels_to_cal = [x for x in range(num_ner)]
    labels_to_cal.remove(index_out)
    label_all = []
    prediction_all = []
    loss_all = []
    loss_all_ce = []
    loss_func = torch.nn.CrossEntropyLoss()
    for step, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        try:
            ouput = model(data)
        except:
            continue
        prediction = ouput['path']
        label = data['label_'+ann_type]
        loss_ce = loss_func(ouput['output'].view(-1, num_ner),
                         data['label_' + ann_type].view(-1))
        loss_all_ce.append(loss_ce.item())
        b_s = len(label)
        for b_s_index in range(b_s):
            prediction_cell = prediction[b_s_index]
            label_cell = label[b_s_index].to('cpu').tolist()
            label_all += label_cell
            prediction_all += prediction_cell

    print('p', precision_score(y_true=label_all, y_pred=prediction_all,
                                         average='micro', labels=labels_to_cal))
    print('r', recall_score(y_true=label_all, y_pred=prediction_all,
                               average='micro', labels=labels_to_cal))
    f = f1_score(y_true=label_all, y_pred=prediction_all,
                 average='micro', labels=labels_to_cal)
    print('f', f)
    if f >= 0.99:
        try:
            torch.save(model.state_dict(),
                       'model_zoo/' + lang + '_' + model_type +
                       str(epoch_num) + '_' +str(round(f * 100, 2)) + '.pth')
        except:
            print('no')
    if lang in ['fre', ]:
        convert_to_tsv(lang, prediction_all, ann_type, epoch_num, f)
    return {'f': f,
            'loss_ce': sum(loss_all_ce) / len(loss_all_ce)}

def convert_to_tsv(lang, prediction, ann_type, epoch_num, f):
    dict_lang_tsv_path = {'fre': 'data/fr_gt.tsv',}
    dict_index_entity_type = {0: 'O', 1: 'B-PER', 2: 'I-PER',
                              3: 'B-ORG', 4: 'I-ORG', 5: 'B-LOC',
                              6: 'I-LOC', 7: 'B-HumanProd', 8: 'I-HumanProd'}
    dict_ann_clumn = {'croase': 'NE-COARSE-LIT'}
    tsv = pd.read_csv(dict_lang_tsv_path[lang])
    type_predict = [dict_index_entity_type[v] for v in prediction[:len(tsv)]]
    tsv[dict_ann_clumn[ann_type]] = type_predict
    tsv.to_csv('log/' + lang + '_' +
                str(epoch_num) + '_' +str(round(f * 100, 2)) + '.tsv',
               index=False)








