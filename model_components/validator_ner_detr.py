import torch
from model_components.loss_func import HungaryLoss
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

def validate(dataloader, model, ann_type, index_out, num_ner):
    labels_to_cal = [x for x in range(num_ner)]
    labels_to_cal.remove(index_out)
    label_all = []
    prediction_all = []
    loss_all = []
    loss_func = HungaryLoss()
    for step, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        ouput = model(data)
        loss = loss_func(ouput, data)
        loss_all.append(loss.item())
        prediction = ouput['path']
        label = data['label_'+ann_type]
        b_s = len(label)
        for b_s_index in range(b_s):
            prediction_cell = prediction[b_s_index]
            label_cell = label[b_s_index].tolist()
            label_all += label_cell
            prediction_all += prediction_cell

    print('p', precision_score(y_true=label_all, y_pred=prediction_all,
                                         average='micro', labels=labels_to_cal))
    print('r', recall_score(y_true=label_all, y_pred=prediction_all,
                               average='micro', labels=labels_to_cal))
    print('f', f1_score(y_true=label_all, y_pred=prediction_all,
                               average='micro', labels=labels_to_cal))

    return sum(loss_all) / len(loss_all)



