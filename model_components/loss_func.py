import torch
from scipy.optimize import linear_sum_assignment
__replace_dict__ = {'fre_newseye':{'replace':{1:1, 2:1, 3:2, 4:2, 5:3, 6:3, 7:4, 8:4},
                                   'seg':[1, 3, 5, 7],
                                   'o':0}}

class HungaryLoss(torch.nn.Module):
    def __init__(self, num_ner, device, window_len):
        super(HungaryLoss, self).__init__()
        self.num_ner = num_ner
        self.device = device
        self.win_len = window_len

    def forward(self, pre, label):
        class_pre = pre['class_pre']
        pos_pre = pre['pos_pre']
        class_gt_padded, pos_gt_padded = self.get_label_padded(label, self.num_ner)
        match_result = self.hungray_match(class_gt_padded, pos_gt_padded,
                                          class_pre, pos_pre)
        loss = self.get_loss(match_result, class_gt_padded, pos_gt_padded,
                                          class_pre, pos_pre)
        return loss


    def get_label_reorganized(self, label):
        label = label.tolist()
        class_gt = []
        pos_gt = []
        index_seg = __replace_dict__['fre_newseye']['seg']
        index_o = __replace_dict__['fre_newseye']['o']
        index_replace = __replace_dict__['fre_newseye']['replace']
        b_s = len(label)
        for b_s_index in range(b_s):
            class_gt_one_batch = []
            pos_gt_one_batch = []
            label_one_batch = label[b_s_index]
            first_index = 0
            while first_index < len(label_one_batch):
                if first_index == len(label_one_batch)-1 and \
                        label_one_batch[first_index] != index_o:
                    class_gt_one_batch.append(index_replace[label_one_batch[first_index]])
                    pos_gt_one_batch.append([first_index, first_index])
                    first_index += 1
                elif label_one_batch[first_index] != index_o:
                    class_gt_one_batch.append(index_replace[label_one_batch[first_index]])
                    for second_index in range(first_index+1, len(label_one_batch)):
                        if second_index==len(label_one_batch)-1 and \
                                label_one_batch[second_index]!=index_o:
                            pos_gt_one_batch.append([first_index, second_index])
                            first_index = second_index + 1
                            break
                        elif label_one_batch[second_index] in index_seg or \
                                label_one_batch[second_index]==index_o:
                            pos_gt_one_batch.append([first_index, second_index-1])
                            first_index = second_index
                            break
                else:
                    first_index += 1
            class_gt.append(class_gt_one_batch)
            pos_gt.append(pos_gt_one_batch)

        return class_gt, pos_gt

    def get_label_padded(self, label, num_ner):
        class_gt, pos_gt = self.get_label_reorganized(label)
        class_gt_padded = []
        pos_gt_padded = []
        b_s = len(class_gt)
        for b_s_index in range(b_s):
            class_gt_one_batch = class_gt[b_s_index] + \
                                 [0] * (num_ner-len(class_gt[b_s_index]))
            class_gt_padded.append(class_gt_one_batch)
            pos_gt_one_batch = pos_gt[b_s_index] + \
                               [[0, 0]] * (num_ner-len(class_gt[b_s_index]))
            pos_gt_padded.append(pos_gt_one_batch)

        return torch.tensor(class_gt_padded).to(self.device), \
               torch.tensor(pos_gt_padded, dtype=torch.float).to(self.device)

    def hungray_match(self, class_gt_padded, pos_gt_padded, class_pre, pos_pre):
        match_result = []
        class_loss = self.get_class_loss(class_gt_padded, class_pre)
        pos_loss = self.get_pos_loss(pos_gt_padded, pos_pre, class_gt_padded)
        all_loss = (class_loss + pos_loss).tolist()
        for b_s_index in range(len(all_loss)):
            match_result.append(linear_sum_assignment(all_loss[b_s_index])[1])

        return match_result

    def get_class_loss(self, class_gt_padded, class_pre):
        b_s = len(class_gt_padded)
        loss = torch.zeros((b_s, self.num_ner, self.num_ner))
        for b_s_index in range(b_s):
            class_pre_one_batch = class_pre[b_s_index]
            class_gt_padded_one_batch = class_gt_padded[b_s_index]
            for ner_index_pre in range(self.num_ner):
                for ner_index_gt in range(self.num_ner):
                    if class_gt_padded_one_batch[ner_index_gt] == 0:
                        loss[b_s_index][ner_index_pre][ner_index_gt] = \
                            torch.tensor(0).to(self.device)
                    else:
                        loss[b_s_index][ner_index_pre][ner_index_gt] = \
                            class_pre_one_batch[ner_index_pre][
                                class_gt_padded_one_batch[ner_index_gt]]

        return -2 * loss * self.win_len

    def get_pos_loss(self, pos_gt_padded, pos_pre, class_gt_padded):
        b_s = len(pos_gt_padded)
        loss = torch.zeros((b_s, self.num_ner, self.num_ner))
        for b_s_index in range(b_s):
            pos_pre_one_batch = pos_pre[b_s_index]
            pos_gt_padded_one_batch = pos_gt_padded[b_s_index]
            class_gt_padded_one_batch = class_gt_padded[b_s_index]
            for ner_index_pre in range(self.num_ner):
                for ner_index_gt in range(self.num_ner):
                    if class_gt_padded_one_batch[ner_index_gt] == 0:
                        loss[b_s_index][ner_index_pre][ner_index_gt] = \
                            torch.tensor(0).to(self.device)
                    else:
                        loss[b_s_index][ner_index_pre][ner_index_gt] = \
                        torch.sum(torch.abs(pos_pre_one_batch[ner_index_pre] -
                                            pos_gt_padded_one_batch[ner_index_gt]))

        return loss

    def get_loss(self, match_result, class_gt_padded, pos_gt_padded,
                                          class_pre, pos_pre):
        b_s = len(match_result)
        class_loss = []
        pos_loss = []
        for b_s_index in range(b_s):
            match_result_one_batch = match_result[b_s_index]
            class_pre_one_batch = class_pre[b_s_index]
            class_gt_one_batch = class_gt_padded[b_s_index][match_result_one_batch]
            pos_pre_one_batch = pos_pre[b_s_index]
            pos_gt_one_batch = pos_gt_padded[b_s_index][match_result_one_batch]
            for ner_index in range(self.num_ner):
                class_loss.append(-1 * torch.log(class_pre_one_batch[ner_index][
                                      class_gt_one_batch][ner_index]))
                if class_gt_one_batch[ner_index] !=0 :
                    pos_loss.append(torch.sum(
                        torch.abs(pos_pre_one_batch[ner_index]
                                  - pos_gt_one_batch[ner_index])))

        return sum(class_loss) + sum(pos_loss)





















