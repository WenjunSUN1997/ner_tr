#  :fire: LIT :fire: : Label-Informed Transformers on Token- and Sequence-based Classification

## 1. Description

In this repo, we propose LIT, an end-to-end pipeline architecture that integrates Transformer's Encoder-Decoder mechanism with an additional label semantic similarity to token-based classification tasks (i.e., historical Named Entity Recognition (NER), Automatic Term Extraction (ATE)) as well as sequence-based classification tasks (i.e., Sentiment Analysis).

---

## 2. Requirements

Please install all the necessary libraries noted in [requirements.txt](./requirements.txt) using this command:

```
pip install -r requirements.txt
```

## 3. Data

First trans your data into a .csv file in this format:
```
words, ner_c, ner_f
['a', ...], [0,1,2...], [0,1,2...]
```
Then add data into /data/
Register the dataset in model/components/dataloader_ner_tr.py and train_ner_tr.py (--lang)

## 4. Implementation

Run the following command to train the model.
```python
python -u train_ner_tr.py
--lang sv
--train_bert 1
--num_encoder 1
--weight_o 15
--sim_dim 768
--batch_size 4
--step_len 50
--window_len 50
--max_len_tokens 190
--data_aug 0
--device cuda:3
--bert_model_name (sv->peanutacake/autotrain-historic-sv-51079121359, fi->peanutacake/autotrain-historic-fi-51081121368)
--model_type ner_tr
--num_ner 13
--sim_dim 768
--ann_type croase
```
## 5. Results

The full results are available later in our paper.

