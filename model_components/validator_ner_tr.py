from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
from model_components.loss_func import FocalLoss

def validate(dataloader, model, ann_type, index_out,
             num_ner, lang, epoch_num, model_type):
    labels_to_cal = [x for x in range(num_ner)]
    labels_to_cal.remove(index_out)
    label_all = []
    prediction_all = []
    loss_all = []
    loss_all_ce = []
    loss_func = torch.nn.CrossEntropyLoss()
    loss_func_ner = FocalLoss(gamma=2.0, alpha=0.9999)
    # loss_func_ner = torch.nn.CrossEntropyLoss()
    for step, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        try:
            ouput = model(data)
        except:
            print('error')
        prediction = ouput['path']
        if model_type == 'ner_tr':
            label = data['label_'+ann_type]
            loss_ce = loss_func(ouput['output'].view(-1, num_ner),
                             data['label_' + ann_type].view(-1))
        else:
            label = data['label_detect']
            loss_ce = ouput['loss']
            # loss_ce = loss_func_ner(ouput['ner_prob'].view(-1, 2),
            #                     data['label_detect'].view(-1))
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
    if f >= 0:
        try:
            # torch.save(model.bert_model.state_dict(),
            #            'model_zoo/' + lang + '_' + model_type +
            #            str(epoch_num) + '_' +str(round(f * 100, 2)) + '.pth')
            torch.save(model.bert_model.state_dict(),
                       'model_zoo/' + lang + '_' + model_type +
                       str(epoch_num) + '_' + str(round(f * 100, 2)) + '.pth')
        except:
            print('no')

    return {'f': f,
            'loss_ce': sum(loss_all_ce) / len(loss_all_ce)}



