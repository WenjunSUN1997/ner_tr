from torch.utils.data import DistributedSampler
from torch.utils.data.dataloader import DataLoader
from model_components.dataset_hipe_bulk import TextDatasetBulk, TextDatasetBulkByLabel
import pandas as pd

def get_dataloader(lang, goal, window_len, step_len, max_len_tokens,
                   tokenizer_name, batch_size, device, num_ner, model_type,
                   data_aug, ann_type):
    file_path_dict = {'fre': {'train': 'data/train_fr.csv',
                              'test': 'data/test_fr.csv',
                              'dev': 'data/test_fr.csv'},
                      'de': {'train': 'data/train_de.csv',
                             'test': 'data/test_de.csv',
                             'dev': 'data/test_de.csv'},
                      'fi': {'train': 'data/train_fi.csv',
                             'test': 'data/test_fi.csv',
                             'dev': 'data/test_fi.csv'},
                      'sv': {'train': 'data/train_sv.csv',
                             'test': 'data/test_sv.csv',
                             'dev': 'data/test_sv.csv'},
                      'wnut': {'train': 'data/train_wnut_17.csv',
                               'test': 'data/test_wnut_17.csv',
                               'dev': 'data/test_wnut_17.csv'},
                      'conll': {'train': 'data/train_conll2003.csv',
                                'test': 'data/test_conll2003.csv',
                                'dev': 'data/test_conll2003.csv'},
                      'conll_au': {'train': 'data/train_conll_augmentation.csv',
                                   'test': 'data/test_conll2003.csv',
                                   'dev': 'data/test_conll2003.csv'},
                      'ajmc_en': {'train': 'data/train_ajmc_en.csv',
                                'test': 'data/test_ajmc_en.csv',
                                'dev': 'data/test_ajmc_en.csv'},
                      'ajmc_fr': {'train': 'data/train_ajmc_fr.csv',
                                  'test': 'data/test_ajmc_fr.csv',
                                  'dev': 'data/test_ajmc_fr.csv'},
                      'ajmc_de': {'train': 'data/train_ajmc_de.csv',
                                  'test': 'data/test_ajmc_de.csv',
                                  'dev': 'data/test_ajmc_de.csv'},
                      '19_en': {'train': 'data/train_19_en.csv',
                                  'test': 'data/test_19_en.csv',
                                  'dev': 'data/test_19_en.csv'},
                      '20_en': {'train': 'data/train_20_en.csv',
                                'test': 'data/test_20_en.csv',
                                'dev': 'data/test_20_en.csv'},
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
                                         model_type=model_type,
                                         ann_type=ann_type)
    elif model_type == 'benchmark':
        dataset = TextDatasetBulk(csv=csv,
                                  window_len=window_len,
                                  step_len=step_len,
                                  device=device,
                                  tokenizer_name=tokenizer_name,
                                  max_len_tokens=max_len_tokens,
                                  goal=goal,
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

    # sampler = DistributedSampler(dataset, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader