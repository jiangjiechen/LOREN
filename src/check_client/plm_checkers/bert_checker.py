# -*- coding: utf-8 -*-

'''
@Author     : Jiangjie Chen
@Time       : 2020/8/18 14:40
@Contact    : jjchen19@fudan.edu.cn
@Description:
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel, BertPreTrainedModel
from .checker_utils import attention_mask_to_mask, ClassificationHead, soft_logic, build_pseudo_labels, \
    get_label_embeddings, temperature_annealing


class BertChecker(BertPreTrainedModel):
    def __init__(self, config, logic_lambda=0.0, prior='nli', m=8, temperature=1):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self._lambda = logic_lambda
        self.prior = prior
        self.temperature = temperature
        self._step = 0

        # general attention
        self.linear_self_attn = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_m_attn = nn.Linear(self.hidden_size * 2, 1, bias=False)

        self.var_hidden_size = self.hidden_size // 4

        z_hid_size = self.num_labels * m
        self.linear_P_theta = nn.Linear(self.hidden_size * 2 + z_hid_size, self.var_hidden_size)
        y_hid_size = self.var_hidden_size
        self.linear_Q_phi = nn.Linear(self.hidden_size * 2 + y_hid_size, self.var_hidden_size)
        
        self.classifier = ClassificationHead(self.var_hidden_size, self.num_labels, config.hidden_dropout_prob) # label embedding for y
        self.z_clf = self.classifier
        self.init_weights()

    def forward(self, claim_input_ids, claim_attention_mask, claim_token_type_ids,
                qa_input_ids_list, qa_attention_mask_list, qa_token_type_ids_list,
                nli_labels=None, labels=None):
        '''
        m: num of questions; n: num of evidence; k: num of candidate answers
        :param claim_input_ids: b x L1
        :param claim_attention_mask: b x L1
        :param claim_token_type_ids: b x L1
        :param qa_input_ids_list: b x m x L2
        :param qa_attention_mask_list: b x m x L2
        :param qa_token_type_ids_list: b x m x L2
        :param labels: (b,)
        :return:
        '''
        self._step += 1
        _zero = torch.tensor(0.).to(claim_input_ids.device)

        global_output = self.bert(
            claim_input_ids,
            attention_mask=claim_attention_mask,
            token_type_ids=claim_token_type_ids
        )[0]  # b x L1 x h

        global_output = self.self_select(global_output)  # b x h

        _qa_input_ids_list = qa_input_ids_list.transpose(1, 0) # m x b x L2
        _qa_attention_mask_list = qa_attention_mask_list.transpose(1, 0)
        _qa_token_type_ids_list = qa_token_type_ids_list.transpose(1, 0)

        local_output_list = []
        for _inp, _attn, _token_ids in zip(_qa_input_ids_list, _qa_attention_mask_list, _qa_token_type_ids_list):
            _local_output = self.bert(_inp, attention_mask=_attn,
                                      token_type_ids=_token_ids)[0]
            _local_output = self.self_select(_local_output)
            local_output_list.append(_local_output)

        local_outputs = torch.stack(local_output_list, 0) # m x b x h
        local_outputs = local_outputs.transpose(1, 0).contiguous() # b x m x h

        neg_elbo, loss, logic_loss = _zero, _zero, _zero
        mask = attention_mask_to_mask(qa_attention_mask_list)
        # b x h, b x m x h -> b x h
        local_outputs_w, m_attn = self.local_attn(global_output, local_outputs, mask)
        local_outputs = torch.cat([local_outputs, global_output.unsqueeze(1).repeat(1, local_outputs.size(1), 1)], -1)

        if labels is not None:
            # Training
            # ======================== Q_phi ================================

            labels_onehot = F.one_hot(labels, num_classes=self.num_labels).to(torch.float)
            y_star_emb = get_label_embeddings(labels_onehot, self.classifier.out_proj.weight)  # b x h
            z = self.Q_phi(local_outputs, y_star_emb)
            z_softmax = z.softmax(-1)

            # ======================== P_theta ==============================

            z_gumbel = F.gumbel_softmax(z, tau=temperature_annealing(self.temperature, self._step),
                                        dim=-1, hard=True)  # b x m x 3
            y = self.P_theta(global_output, local_outputs_w, z_gumbel)

            # ======================== soft logic ===========================
            mask = mask.to(torch.int)
            y_z = soft_logic(z_softmax, mask)  # b x 3
            logic_loss = F.kl_div(y.log_softmax(-1), y_z)

            # ======================== ELBO =================================
            elbo_neg_p_log = F.cross_entropy(y.view(-1, self.num_labels), labels.view(-1))
            if self.prior == 'nli':
                prior = nli_labels.softmax(dim=-1)
            elif self.prior == 'uniform':
                prior = torch.tensor([1 / self.num_labels] * self.num_labels).to(y)
                prior = prior.unsqueeze(0).unsqueeze(0).repeat(mask.size(0), mask.size(1), 1)
            elif self.prior == 'logic':
                prior = build_pseudo_labels(labels, m_attn)
            else:
                raise NotImplementedError(self.prior)

            elbo_kl = F.kl_div(z_softmax.log(), prior)
            neg_elbo = elbo_kl + elbo_neg_p_log

            loss = (1 - abs(self._lambda)) * neg_elbo + abs(self._lambda) * logic_loss
        else:
            # Inference
            if self.prior == 'nli':
                z = nli_labels
            elif self.prior == 'uniform':
                prior = torch.tensor([1 / self.num_labels] * self.num_labels).to(y)
                z = prior.unsqueeze(0).unsqueeze(0).repeat(mask.size(0), mask.size(1), 1)
            else:
                z = torch.rand([local_outputs.size(0), local_outputs.size(1), self.num_labels]).to(local_outputs)
            z_softmax = z.softmax(-1)

            for i in range(3):  # N = 3
                z = z_softmax.argmax(-1)
                z = F.one_hot(z, num_classes=3).to(torch.float)
                y = self.P_theta(global_output, local_outputs_w, z)
                y = y.softmax(-1)
                y_emb = get_label_embeddings(y, self.classifier.out_proj.weight)
                z = self.Q_phi(local_outputs, y_emb)
                z_softmax = z.softmax(-1)

        return (loss, (neg_elbo, logic_loss), y, m_attn, (z_softmax, mask))  # batch first

    def Q_phi(self, X, y):
        '''
        X, y => z
        :param X: b x m x h
        :param y_emb: b x 3 / b x h'
        :return: b x m x 3 (ref, nei, sup)
        '''
        y_expand = y.unsqueeze(1).repeat(1, X.size(1), 1)   # b x m x 3/h'
        z_hidden = self.linear_Q_phi(torch.cat([y_expand, X], dim=-1))  # b x m x h'
        z_hidden = F.tanh(z_hidden)
        z = self.z_clf(z_hidden)
        return z

    def P_theta(self, X_global, X_local, z):
        '''
        X, z => y*
        :param X_global: b x h
        :param X_local: b x m x h
        :param z: b x m x 3
        :param mask: b x m
        :return: b x 3, b x m
        '''
        b = z.size(0)
        # global classification
        _logits = torch.cat([X_local, X_global, z.reshape(b, -1)], dim=-1)
        _logits = self.dropout(_logits)
        _logits = self.linear_P_theta(_logits)
        _logits = torch.tanh(_logits)
        
        y = self.classifier(_logits)
        return y

    def self_select(self, h_x):
        '''
        self attention on a vector
        :param h_x: b x L x h
        :return: b x h
        '''
        w = self.dropout(self.linear_self_attn(h_x).squeeze(-1)).softmax(-1)
        return torch.einsum('blh,bl->bh', h_x, w)

    def local_attn(self, global_output, local_outputs, mask):
        '''
        :param global_output: b x h
        :param qa_outputs: b x m x h
        :param mask: b x m
        :return: b x h, b x m
        '''
        m = local_outputs.size(1)
        scores = self.linear_m_attn(torch.cat([global_output.unsqueeze(1).repeat(1, m, 1),
                                               local_outputs], dim=-1)).squeeze(-1) # b x m
        mask = 1 - mask
        scores = scores.masked_fill(mask.to(torch.bool), -1e16)
        attn = F.softmax(scores, -1)
        return torch.einsum('bm,bmh->bh', attn, local_outputs), attn
