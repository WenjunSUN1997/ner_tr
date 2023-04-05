import torch
__replace_dict__ = {'fre_newseye':{'replace':{1:1, 2:1, 3:2, 4:2, 5:3, 6:3, 7:4, 8:4},
                                   'seg':[1, 3, 5, 7],
                                   'o':0}}
def post_process(output, window_len, dataset_name):
    class_pre = output['class_pre']
    pos_pre = torch.round(output['pos_pre'])
    b_s = pos_pre.shape[0]
    pre = []
    max_val, max_idx = torch.max(class_pre, dim=-1)
    for b_s_index in range(b_s):
        class_one_batch = max_idx[b_s_index]
        pos_one_batch = pos_pre[b_s_index]
        pre_one_batch = [0] * window_len
        for ner_index in range(len(class_one_batch)):
            if class_one_batch[ner_index] == 0:
                continue
            else:
                result = [key for key, value in
                          __replace_dict__[dataset_name]['replace'].items()
                          if value == int(class_one_batch[ner_index].item())]
                result.sort()
                start, length  = pos_one_batch[ner_index]
                if start + length >= window_len:
                    length = window_len-1-start
                pre_one_batch[int(start.item()):int(start.item()+length.item())+1] = \
                    [result[1]] * (int(length.item()+1))
                pre_one_batch[int(start.item())] = result[0]
        pre.append(pre_one_batch)

    return pre







