# ner_tr
Finally someone is watching, I thought we could all communicate telepathically, but the reality is harsh.  
But, since you asked in good faith, I'll answer you properly!


![屏幕截图 2023-04-27 174803](https://user-images.githubusercontent.com/20592128/234916412-d5b3bac4-ec49-4e97-8ccc-19381e64b682.png)

first trans your data into this format:  
words, ner_c, ner_f  
['a', ...], [0,1,2...], [0,1,2...]

then add your data into /data/   
Register your dataset in model/components/dataloader_ner_tr.py and train_ner_tr.py (--lang)  
Run  

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
