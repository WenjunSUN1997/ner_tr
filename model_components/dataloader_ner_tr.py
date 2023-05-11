from torch.utils.data.dataloader import DataLoader
from model_components.dataset_hipe_bulk import TextDatasetBulk, TextDatasetBulkByLabel
import pandas as pd

def get_dataloader(lang, goal, window_len, step_len, max_len_tokens,
                   tokenizer_name, batch_size, device, num_ner, model_type,
                   data_aug):
    file_path_dict = {'fre': {'train': 'data/train_fr.csv',
                              'test': 'data/test_fr.csv',
                              'dev': 'data/test_fr.csv'},
                      'wnut': {'train': 'data/train_wnut_17.csv',
                               'test': 'data/test_wnut_17.csv',
                               'dev': 'data/test_wnut_17.csv'},
                      'conll': {'train': 'data/train_conll2003.csv',
                                'test': 'data/test_conll2003.csv',
                                'dev': 'data/test_conll2003.csv'},
                      'conll_au': {'train': 'data/train_conll_augmentation.csv',
                                   'test': 'data/test_conll2003.csv',
                                   'dev': 'data/test_conll2003.csv'},
                      'phoner': {'train': 'data/train_syllable_vi.csv',
                                   'test': 'data/dev_syllable_vi.csv',
                                   'dev': 'data/test_syllable_vi.csv'},
                      'en_ate': {'train': 'data/ate/en/en_ann_corp.csv',
                                   'test': 'data/ate/en/en_ann_equi.csv',
                                   'dev': 'data/ate/en/en_ann_htfl.csv'},
                      }
    if data_aug:
        file_path_dict['fre']['train'] = 'data/train_fr_augmentation.csv'

    csv = pd.read_csv(file_path_dict[lang][goal])

    if goal == 'train' and model_type == 'detector':
        dataset = TextDatasetBulk(csv=csv,
                                  window_len=window_len,
                                  step_len=step_len,
                                  device=device,
                                  tokenizer_name=tokenizer_name,
                                  max_len_tokens=max_len_tokens,
                                  goal=goal,
                                  model_type=model_type)
    elif goal == 'train' and model_type == 'ner_tr':
        dataset = TextDatasetBulkByLabel(csv=csv,
                                  window_len=window_len,
                                  step_len=step_len,
                                  device=device,
                                  tokenizer_name=tokenizer_name,
                                  max_len_tokens=max_len_tokens,
                                  goal=goal,
                                  num_ner=num_ner,
                                  model_type=model_type)
    else:
        dataset = TextDatasetBulk(csv=csv,
                                  window_len=window_len,
                                  step_len=step_len,
                                  device=device,
                                  tokenizer_name=tokenizer_name,
                                  max_len_tokens=max_len_tokens,
                                  goal=goal,
                                  model_type=model_type)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader