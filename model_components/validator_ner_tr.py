from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
from sklearn.metrics import confusion_matrix
import pandas as pd
import os

def validate(dataloader, model, ann_type, index_out,
             num_ner, lang, epoch_num, model_type,
             loss_func):
    labels_to_cal = [x for x in range(num_ner)]
    labels_to_cal.remove(index_out)
    label_all = []
    prediction_all = []
    loss_all_ce = []
    tokens = []
    for step, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        batch_size = data['input_ids'].shape[0]
        for batch_size_index in range(batch_size):
            tokens += [x[batch_size_index] for x in data['original_token']]
        ouput = model(data)
        prediction = ouput['path']
        label = data['label_'+ann_type]
        if model_type != 'benchmark':
            loss_ce = loss_func(ouput['output'].view(-1, num_ner),
                                data['label_' + ann_type].view(-1))
        else:
            loss_ce = ouput['loss']

        loss_all_ce.append(loss_ce.item())
        b_s = len(label)
        for b_s_index in range(b_s):
            prediction_cell = prediction[b_s_index]
            label_cell = label[b_s_index].to('cpu').tolist()
            label_all += label_cell
            prediction_all += prediction_cell
    if 'ann' in lang or 'nes' in lang:
        extracted_terms = get_term(label_list=prediction_all, token_list=tokens)
        truth = get_term(label_list=label_all, token_list=tokens)
        p, r, f = computeTermEvalMetrics(extracted_terms=extracted_terms,
                                         gold_df=truth)
        record_term(extracted_terms=extracted_terms, gold_df=truth,
                    f=f, lang=lang, epoch_num=epoch_num)

    else:
        p = precision_score(y_true=label_all, y_pred=prediction_all,
                            average='micro', labels=labels_to_cal)
        r = recall_score(y_true=label_all, y_pred=prediction_all,
                         average='micro', labels=labels_to_cal)
        f = f1_score(y_true=label_all, y_pred=prediction_all,
                 average='micro', labels=labels_to_cal)
    c_m = confusion_matrix(label_all, prediction_all)
    print('p', p)
    print('r', r)
    print('f', f)
    for row in c_m:
        row_str = ""
        for num in row:
            row_str += f"{num:5d} "
        print(row_str)

    if f >= 0.99:
        torch.save(model.state_dict(),
                   'model_zoo/' + lang + '_' + model_type +
                   str(epoch_num) + '_' +str(round(f * 100, 2)) + '.pth')
    try:
        convert_tsv(lang, prediction_all, epoch_num, f)
    except:
        print('can not store result')

    return {'f': f,
            'p': p,
            'r': r,
            'loss_ce': sum(loss_all_ce) / len(loss_all_ce)}

def convert_tsv(lang, prediction_all, epoch_num, f):
    index_tag_dict = {'fre':['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG',
                              'B-LOC', 'I-LOC', 'B-HumanProd', 'I-HumanProd',
                              'B-PER.author', 'I-PER.author'],
                      'fi': ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG',
                              'B-LOC', 'I-LOC', 'B-HumanProd', 'I-HumanProd',
                              'B-PER.author', 'I-PER.author'],
                      'sv': ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG',
                              'B-LOC', 'I-LOC', 'B-HumanProd', 'I-HumanProd',
                              'B-PER.author', 'I-PER.author'],
                      '20_en': ['O','B-prod','I-prod','B-loc','I-loc','B-org','I-org',
                                'B-time','I-time','B-pers','I-pers'],
                      '19_en': ['O','B-LOC','I-LOC','B-BUILDING','I-BUILDING','B-STREET','I-STREET'],
                      'ajmc_en': ['O','B-pers','I-pers',
                                  'B-work','I-work',
                                  'B-scope','I-scope',
                                  'B-loc','I-loc',
                                  'B-date','I-date',
                                  'B-object','I-object'],
                      }
    gt_lang_dict = {'fre': 'data/fr_gt.tsv',
                    'fi': 'data/gt_fi.tsv',
                    'sv': 'data/gt_sv.tsv',
                    '20_en': 'data/gt_20_en.tsv',
                    '19_en': 'data/gt_19_en.tsv',
                    'ajmc_en': 'data/gt_ajmc_en.tsv',
                    }
    gt = pd.read_csv(gt_lang_dict[lang])
    ner_c = [index_tag_dict[lang][v] for v in prediction_all]
    gt['NE-COARSE-LIT'] = ner_c[:len(gt['TOKEN'])]
    folder = os.path.exists('log/'+lang)
    if not folder:
        os.makedirs('log/'+lang)

    gt.to_csv('log/' + lang + '/' + lang + '_' + str(epoch_num)
              + '_' + str(round(f * 100, 2)) + '.tsv',
              index=False)

def get_term(label_list, token_list):
    term_list = []
    for index in range(len(label_list)):
        term = ''
        if label_list[index] == 1:
            term += token_list[index]
            for index_2 in range(index+1, len(label_list)):
                if label_list[index_2] == 2:
                    term += token_list[index_2] + ' '
                else:
                    break
        term_list.append(term)

    return set(term_list)

def computeTermEvalMetrics(extracted_terms, gold_df):
    #make lower case cause gold standard is lower case
    extracted_terms = set([item.lower() for item in extracted_terms])
    gold_set = set(gold_df)
    true_pos = extracted_terms.intersection(gold_set)
    recall = round(len(true_pos)*100/len(gold_set),2)
    precision = round(len(true_pos)*100/len(extracted_terms),2)
    fscore = round(2*(precision*recall)/(precision+recall),2)
    return precision, recall, fscore

def record_term(extracted_terms, gold_df, f, lang, epoch_num):
    folder = os.path.exists('log/' + lang)
    if not folder:
        os.makedirs('log/' + lang)

    with open('log/' + lang + '/' + lang + '_' + str(epoch_num)
              + '_' + str(round(f * 100, 2)) + '.txt', 'w') as file:
        for v in extracted_terms:
            file.write(str(v) + '\n')

    with open('log/' + lang + '/' + lang + '_gold.txt', 'w') as file:
        for v in gold_df:
            file.write(str(v) + '\n')





