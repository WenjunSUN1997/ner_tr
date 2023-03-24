import torch
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

def validate(dataloader, model, ann_type, index_out, num_ner):
    labels_to_cal = [x for x in range(num_ner)]
    labels_to_cal.remove(index_out)
    p_all = []
    r_all = []
    f_all = []
    label_all = []
    prediction_all = []
    loss_all = []
    for data in tqdm(dataloader):
        ouput = model(data)
        loss = ouput['loss']
        loss_all.append(loss.item())
        prediction = ouput['path']
        label = data['label_'+ann_type]
        b_s = len(label)
        for b_s_index in range(b_s):
            prediction_cell = prediction[b_s_index]
            label_cell = label[b_s_index].tolist()
            p_all.append(precision_score(y_true=label_cell, y_pred=prediction_cell,
                                         average='macro', labels=labels_to_cal))
            r_all.append(recall_score(y_true=label_cell, y_pred=prediction_cell,
                                         average='macro', labels=labels_to_cal))
            f_all.append(f1_score(y_true=label_cell, y_pred=prediction_cell,
                                         average='macro', labels=labels_to_cal))
            label_all += label_cell
            prediction_all += prediction_cell

    print('loss', sum(loss_all) / len(loss_all))
    print('p', sum(p_all) / len(p_all))
    print('r', sum(r_all) / len(r_all))
    print('f', sum(f_all) / len(f_all))
    print('p', precision_score(y_true=label_all, y_pred=prediction_all,
                                         average='macro', labels=labels_to_cal))
    print('r', recall_score(y_true=label_all, y_pred=prediction_all,
                               average='macro', labels=labels_to_cal))
    print('f', f1_score(y_true=label_all, y_pred=prediction_all,
                               average='macro', labels=labels_to_cal))

    return sum(loss_all) / len(loss_all)



