# -*- coding: utf-8 -*-

'''
@Author     : Jiangjie Chen
@Time       : 2020/8/12 14:44
@Contact    : jjchen19@fudan.edu.cn
@Description: 
'''

import re
import time
from pathlib import Path
from typing import Dict, List
import torch
from logging import getLogger
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import ujson as json
import random

try:
    from .seq2seq.seq2seq_utils import (
        use_task_specific_params,
        calculate_rouge,
        chunks,
        Seq2SeqDataset,
        lmap,
        load_json,
        save_json,
    )
except ImportError:
    import cjjpy as cjj
    import sys
    sys.path.append(cjj.AbsParentDir(__file__, '.'))
    from seq2seq.seq2seq_utils import (
        use_task_specific_params,
        calculate_rouge,
        chunks,
        Seq2SeqDataset,
        lmap,
        load_json,
        save_json,
    )

logger = getLogger(__name__)
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(1111)


def assemble_answers_to_one(js, k=5, mask_token='<mask>', mask_rate=0.):
    if isinstance(js, str):
        js = json.loads(js)

    should_keep = random.random() > mask_rate
    if 'evidential_assembled' in js:
        js.pop('evidential_assembled')
    for q, answers in zip(js['cloze_qs'], js['evidential']):
        if mask_token in q:
            s = q.find(mask_token)
            e = s + len(mask_token)
            nq_list = []
            if should_keep:
                for i in range(k):
                    answer_span = answers[i]
                    nq = q[:s] + answer_span + q[e:]
                    nq_list.append(nq)
            else:
                for i in range(k):
                    answer_span = mask_token
                    nq = q[:s] + answer_span + q[e:]
                    nq_list.append(nq)
            ev_nqs = ' '.join(nq_list)
            if js.get('evidential_assembled') is None:
                js['evidential_assembled'] = [ev_nqs]
            else:
                js['evidential_assembled'].append(ev_nqs)
    assert len(js['evidential_assembled']) == len(js['answers'])
    return js


class AnswerGenerator():
    def __init__(self, model_name, device=DEFAULT_DEVICE):
        self.model_name = str(model_name)
        self.device = device
        self.model = None
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def init_model(self):
        if self.model is None:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)

    def assemble(self, question, context):
        sep = '\n' if 'unifiedqa' in self.tokenizer.name_or_path else self.tokenizer.sep_token
        return f'{question} {sep} {context}'

    def generate(self, examples, out_file=None, batch_size=16, verbose=True,
                 max_length=20, min_length=1, num_beams=4, num_return_sequences=4,
                 prefix=None, fp16=False, task='summarization', **generate_kwargs):
        '''
        :param examples: [N]
        :return: [N x num_return_seq]
        '''
        self.init_model()
        if fp16:
            self.model = self.model.half()
        # update config with summarization specific params
        use_task_specific_params(self.model, task)

        fout = None if out_file is None else Path(out_file).open("w", encoding="utf-8")
        generated = []
        if verbose:
            iter = tqdm(list(chunks(examples, batch_size)), desc="MRC")
        else:
            iter = list(chunks(examples, batch_size))
        if prefix is None:
            prefix = prefix or getattr(self.model.config, "prefix", "") or ""
        for examples_chunk in iter:
            examples_chunk = [prefix + text for text in examples_chunk]
            batch = self.tokenizer(examples_chunk, return_tensors="pt", truncation=True,
                                   padding="longest").to(self.device)
            summaries = self.model.generate(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                length_penalty=1.2,
                repetition_penalty=1.2,
                **generate_kwargs,
            )
            dec = self.tokenizer.batch_decode(summaries, skip_special_tokens=True,
                                              clean_up_tokenization_spaces=False)
            if fout is not None:
                for hypothesis in dec:
                    fout.write(hypothesis.strip() + "\n")
                    fout.flush()
            else:
                generated += dec
        if fout is not None:
            fout.close()
        generated = list(map(lambda x: x.strip(), generated))
        generated = list(chunks(generated, num_return_sequences))
        return generated

