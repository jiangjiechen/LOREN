# -*- coding: utf-8 -*-

'''
@Author     : Jiangjie Chen
@Time       : 2020/10/15 16:10
@Contact    : jjchen19@fudan.edu.cn
@Description: 
'''

import torch
import random
import torch.nn.functional as F
import torch.nn as nn


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, hidden_size, num_labels, hidden_dropout_prob=0.2):
        super().__init__()
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.out_proj = nn.Linear(hidden_size, num_labels, bias=False)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


def temperature_annealing(tau, step):
    if tau == 0.:
        tau = 10. if step % 5 == 0 else 1.
    return tau


def get_label_embeddings(labels, label_embedding):
    '''
    :param labels: b x 3
    :param label_embedding: 3 x h'
    :return: b x h'
    '''
    emb = torch.einsum('oi,bo->bi', label_embedding, labels)
    return emb


def soft_logic(y_i, mask, tnorm='product'):
    '''
    a^b = ab
    avb = 1 - ((1-a)(1-b))
    :param y_i: b x m x 3
    :param mask: b x m
    :param tnorm: product or godel or lukasiewicz
    :return: [b x 3]
    '''
    _sup = y_i[:, :, 2]  # b x m
    _ref = y_i[:, :, 0]  # b x m
    _sup = _sup * mask + (1 - mask)  # pppp1111
    _ref = _ref * mask  # pppp0000

    if tnorm == 'product':
        p_sup = torch.exp(torch.log(_sup).sum(1))
        p_ref = 1 - torch.exp(torch.log(1 - _ref).sum(1))
    elif tnorm == 'godel':
        p_sup = _sup.min(-1).values
        p_ref = _ref.max(-1).values
    elif tnorm == 'lukas':
        raise NotImplementedError(tnorm)
    else:
        raise NotImplementedError(tnorm)

    p_nei = 1 - p_sup - p_ref
    p_sup = torch.max(p_sup, torch.zeros_like(p_sup))
    p_ref = torch.max(p_ref, torch.zeros_like(p_ref))
    p_nei = torch.max(p_nei, torch.zeros_like(p_nei))
    logical_prob = torch.stack([p_ref, p_nei, p_sup], dim=-1)
    assert torch.lt(logical_prob, 0).to(torch.int).sum().tolist() == 0, \
        (logical_prob, _sup, _ref)
    return logical_prob  # b x 3


def build_pseudo_labels(labels, m_attn):
    '''
    :param labels: (b,)
    :param m_attn: b x m
    :return: b x m x 3
    '''
    mask = torch.gt(m_attn, 1e-16).to(torch.int)
    sup_label = torch.tensor(2).to(labels)
    nei_label = torch.tensor(1).to(labels)
    ref_label = torch.tensor(0).to(labels)
    pseudo_labels = []
    for idx, label in enumerate(labels):
        mm = mask[idx].sum(0)
        if label == 2: # SUPPORTS
            pseudo_label = F.one_hot(sup_label.repeat(mask.size(1)), num_classes=3).to(torch.float) # TODO: hyperparam

        elif label == 0: # REFUTES
            num_samples = magic_proportion(mm)
            ids = torch.topk(m_attn[idx], k=num_samples).indices
            pseudo_label = []
            for i in range(mask.size(1)):
                if i >= mm:
                    _label = torch.tensor([1/3, 1/3, 1/3]).to(labels)
                elif i in ids:
                    _label = F.one_hot(ref_label, num_classes=3).to(torch.float)
                else:
                    if random.random() > 0.5:
                        _label = torch.tensor([0., 0., 1.]).to(labels)
                    else:
                        _label = torch.tensor([0., 1., 0.]).to(labels)
                pseudo_label.append(_label)
            pseudo_label = torch.stack(pseudo_label)

        else: # NEI
            num_samples = magic_proportion(mm)
            ids = torch.topk(m_attn[idx], k=num_samples).indices
            pseudo_label = sup_label.repeat(mask.size(1))
            pseudo_label[ids] = nei_label
            pseudo_label = F.one_hot(pseudo_label, num_classes=3).to(torch.float) # TODO: hyperparam
        
        pseudo_labels.append(pseudo_label)
    return torch.stack(pseudo_labels)


def magic_proportion(m, magic_n=5):
    # 1~4: 1, 5~m: 2
    return m // magic_n + 1


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


def collapse_w_mask(inputs, mask):
    '''
    :param inputs: b x L x h
    :param mask: b x L
    :return: b x h
    '''
    hidden = inputs.size(-1)
    output = inputs * mask.unsqueeze(-1).repeat((1, 1, hidden))  # b x L x h
    output = output.sum(-2)
    output /= (mask.sum(-1) + 1e-6).unsqueeze(-1).repeat((1, hidden))  # b x h
    return output


def parse_ce_outputs(ce_seq_output, ce_lengths):
    '''
    :param qa_seq_output: b x L1 x h
    :param qa_lengths: e.g. [0,1,1,0,2,2,0,0] (b x L2)
    :return:
        c_output: b x h
        e_output: b x h
    '''
    if ce_lengths.max() == 0:
        b, L1, h = ce_seq_output.size()
        return torch.zeros([b, h]).cuda(), torch.zeros([b, h]).cuda()
    masks = []
    for mask_id in range(1, ce_lengths.max() + 1):
        _m = torch.ones_like(ce_lengths) * mask_id
        mask = _m.eq(ce_lengths).to(torch.int)
        masks.append(mask)
    c_output = collapse_w_mask(ce_seq_output, masks[0])
    e_output = torch.stack([collapse_w_mask(ce_seq_output, m)
                            for m in masks[1:]]).mean(0)
    return c_output, e_output


def parse_qa_outputs(qa_seq_output, qa_lengths, k):
    '''
    :param qa_seq_output: b x L2 x h
    :param qa_lengths: e.g. [0,1,1,0,2,2,0,3,0,4,0,5,0,0,0,0] (b x L2)
    :return:
        q_output: b x h
        a_output: b x h
        k_cand_output: k x b x h
    '''
    b, L2, h = qa_seq_output.size()
    if qa_lengths.max() == 0:
        return torch.zeros([b, h]).cuda(), torch.zeros([b, h]).cuda(), \
               torch.zeros([k, b, h]).cuda()

    masks = []
    for mask_id in range(1, qa_lengths.max() + 1):
        _m = torch.ones_like(qa_lengths) * mask_id
        mask = _m.eq(qa_lengths).to(torch.int)
        masks.append(mask)

    q_output = collapse_w_mask(qa_seq_output, masks[0])
    a_output = collapse_w_mask(qa_seq_output, masks[1])
    k_cand_output = [collapse_w_mask(qa_seq_output, m)
                     for m in masks[2:2 + k]]
    for i in range(k - len(k_cand_output)):
        k_cand_output.append(torch.zeros([b, h]).cuda())
    k_cand_output = torch.stack(k_cand_output, dim=0)

    return q_output, a_output, k_cand_output


def attention_mask_to_mask(attention_mask):
    '''
    :param attention_mask: b x m x L
    :return: b x m
    '''
    mask = torch.gt(attention_mask.sum(-1), 0).to(torch.int).sum(-1)  # (b,)
    mask = sequence_mask(mask, max_len=attention_mask.size(1)).to(torch.int)  # (b, m)
    return mask


if __name__ == "__main__":
    y = torch.tensor([[[0.3,0.5,0.2],[0.1,0.4,0.5]]])
    mask = torch.tensor([1,1])
    s = soft_logic(y, mask)
    print(s)