from torch.utils.data.dataloader import DataLoader
from model_components.dataset_hipe import TextDataset
import pandas as pd

def get_dataloader(lang, goal, max_len_words, max_len_tokens,
                   tokenizer_name, batch_size, device):
    file_path_dict = {'fre':{'train':'data/train_fr.csv', 'test':'data/test_fr.csv'}}
    csv = pd.read_csv(file_path_dict[lang][goal])

    dataset = TextDataset(csv=csv,
                          max_len_words=max_len_words,
                          max_len_tokens=max_len_tokens,
                          tokenizer_name=tokenizer_name,
                          device=device)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader