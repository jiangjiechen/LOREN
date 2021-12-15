# -*- coding: utf-8 -*-

"""
@Author     : Jiangjie Chen
@Time       : 2021/5/7 19:39
@Contact    : jjchen19@fudan.edu.cn
@Description:
"""

import sys
import os
import cjjpy as cjj
from tqdm import tqdm
import ujson as json
import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from ..mrc_client.answer_generator import assemble_answers_to_one, chunks
except:
    sys.path.append('..')
    from mrc_client.answer_generator import assemble_answers_to_one, chunks


def load_model(model_name_or_path, device='cuda'):
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return model, tokenizer


def run_nli_line(line, model, tokenizer):
    js = json.loads(line) if isinstance(line, str) else line
    js = assemble_answers_to_one(js, 1)
    premises, hypotheses = [], []
    for ev in js['evidential_assembled']:
        premises.append(ev)
        hypotheses.append(js['claim'])
    nli_labels = []
    for p_chunk, h_chunk in zip(chunks(premises, 8), chunks(hypotheses, 8)):
        inputs = tokenizer(p_chunk, h_chunk, return_tensors='pt', padding=True, truncation=True).to(model.device)
        s = model(**inputs).logits.tolist()
        nli_labels += s
    assert len(nli_labels) == len(js['answers'])
    js['nli_labels'] = nli_labels
    return js


def run(filename, model, tokenizer):
    with open(filename) as f:
        data = f.readlines()
        with open(filename, 'w') as fo:
            for line in tqdm(data, desc=os.path.basename(filename)):
                js = run_nli_line(line, model, tokenizer)
                fo.write(json.dumps(js) + '\n')
    cjj.lark(f'{filename} done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', '-m', type=str, required=True)
    parser.add_argument('--input', '-i', type=str, required=True)
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_name_or_path)
    run(args.input, model, tokenizer)