from torch.utils.data.dataloader import DataLoader
from model_components.dataset_hipe_bulk import TextDatasetBulk, TextDatasetBulkByLabel
import pandas as pd

def get_dataloader(lang, goal, window_len, step_len, max_len_tokens,
                   tokenizer_name, batch_size, device, num_ner):
    file_path_dict = {'fre':{'train':'data/train_fr.csv',
                             'test':'data/test_fr.csv',
                             'dev':'data/dev_fr.csv'},
                      'wnut': {'train': 'data/train_wnut_17.csv',
                              'test': 'data/test_wnut_17.csv',
                              'dev': 'data/test_wnut_17.csv'},
                      'conll': {'train': 'data/train_conll2003.csv',
                               'test': 'data/test_conll2003.csv',
                               'dev': 'data/test_conll2003.csv'}
                      }
    csv = pd.read_csv(file_path_dict[lang][goal])

    if goal == 'train':
        dataset = TextDatasetBulkByLabel(csv=csv,
                                  window_len=window_len,
                                  step_len=step_len,
                                  device=device,
                                  tokenizer_name=tokenizer_name,
                                  max_len_tokens=max_len_tokens,
                                  goal=goal, num_ner=num_ner)

    else:
        dataset = TextDatasetBulk(csv=csv,
                                  window_len=window_len,
                                  step_len=step_len,
                                  device=device,
                                  tokenizer_name=tokenizer_name,
                                  max_len_tokens=max_len_tokens,
                                  goal=goal)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader