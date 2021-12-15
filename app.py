# -*- coding: utf-8 -*-

"""
@Author     : Jiangjie Chen
@Time       : 2021/12/13 17:17
@Contact    : jjchen19@fudan.edu.cn
@Description:
"""

import os
import gradio as gr
from src.loren import Loren
from huggingface_hub import snapshot_download
from prettytable import PrettyTable
import pandas as pd

config = {
    "input": "demo",
    "model_type": "roberta",
    "model_name_or_path": "roberta-large",
    "logic_lambda": 0.5,
    "prior": "random",
    "mask_rate": 0.0,
    "cand_k": 3,
    "max_seq1_length": 256,
    "max_seq2_length": 128,
    "max_num_questions": 8,
    "do_lower_case": False
}

model_dir = snapshot_download('Jiangjie/loren')
# os.makedirs('data/', exist_ok=True)
# os.system('wget -O data/fever.db https://s3-eu-west-1.amazonaws.com/fever.public/wiki_index/fever.db')

config['fc_dir'] = os.path.join(model_dir, 'fact_checking/roberta-large/')
config['mrc_dir'] = os.path.join(model_dir, 'mrc_seq2seq/bart-base/')
config['er_dir'] = os.path.join(model_dir, 'evidence_retrieval/')

loren = Loren(config)
try:
    # js = {
    #     'id': 0,
    #     'evidence': ['EVIDENCE1', 'EVIDENCE2'],
    #     'question': ['QUESTION1', 'QUESTION2'],
    #     'claim_phrase': ['CLAIMPHRASE1', 'CLAIMPHRASE2'],
    #     'local_premise': [['E1 ' * 100, 'E1' * 100, 'E1' * 10], ['E2', 'E2', 'E2']],
    #     'phrase_veracity': [[0.1, 0.5, 0.4], [0.1, 0.7, 0.2]],
    #     'claim_veracity': 'SUPPORT'
    # }
    js = loren.check('Donald Trump won the 2020 U.S. presidential election.')
except Exception as e:
    raise ValueError(e)


def gradio_formatter(js, output_type):
    if output_type == 'e':
        data = {'Evidence': js['evidence']}
    elif output_type == 'z':
        data = {
            'Claim Phrase': js['claim_phrases'],
            'Local Premise': js['local_premises'],
            'p_SUP': [round(x[2], 4) for x in js['phrase_veracity']],
            'p_REF': [round(x[0], 4) for x in js['phrase_veracity']],
            'p_NEI': [round(x[1], 4) for x in js['phrase_veracity']],
        }
    else:
        raise NotImplementedError
    data = pd.DataFrame(data)
    pt = PrettyTable(field_names=list(data.columns))
    for v in data.values:
        pt.add_row(v)

    html = pt.get_html_string(attributes={
        'style': 'border: 1; border-width: 1px; border-collapse: collapse; align: left',
    }, format=True)
    return html


def run(claim):
    js = loren.check(claim)
    ev_html = gradio_formatter(js, 'e')
    z_html = gradio_formatter(js, 'z')
    return ev_html, z_html, js['claim_veracity'], js


iface = gr.Interface(
    fn=run,
    inputs="text",
    outputs=[
        'html',
        'html',
        'label',
        'json'
    ],
    examples=['Donald Trump won the U.S. 2020 presidential election.',
              'The first inauguration of Bill Clinton was in the United States.',
              'The Cry of the Owl is based on a book by an American.',
              'Smriti Mandhana is an Indian woman.'],
    title="LOREN",
    layout='horizontal',
    description="LOREN is an interpretable Fact Verification model against Wikipedia. "
                "This is a demo system for \"LOREN: Logic-Regularized Reasoning for Interpretable Fact Verification\". "
                "See the paper for technical details. You can add FLAG on the bottom to record interesting or bad cases! \n"
                "*Note that the demo system directly retrieves evidence from an up-to-date Wikipedia, which is different from the evidence used in the paper.",
    flagging_dir='results/flagged/',
    allow_flagging=True,
    flagging_options=['Good Case!', 'Error: MRC', 'Error: Parsing',
                      'Error: Commonsense', 'Error: Evidence', 'Error: Other'],
    enable_queue=True
)
iface.launch()
