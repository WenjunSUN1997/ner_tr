from torch.utils.data.dataloader import DataLoader
from model_components.dataset_hipe_bulk import TextDatasetBulk
import pandas as pd

def get_dataloader(lang, goal, window_len, step_len, max_len_tokens,
                   tokenizer_name, batch_size, device):
    file_path_dict = {'fre':{'train':'data/train_fr.csv',
                             'test':'data/test_fr.csv',
                             'dev':'data/dev_fr.csv'}}
    csv = pd.read_csv(file_path_dict[lang][goal])

    dataset = TextDatasetBulk(csv=csv,
                              window_len=window_len,
                              step_len=step_len,
                              device=device,
                              tokenizer_name=tokenizer_name,
                              max_len_tokens=max_len_tokens)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader