# -*- coding: utf-8 -*-

'''
@Author     : Jiangjie Chen
@Time       : 2020/7/29 21:50
@Contact    : jjchen19@fudan.edu.cn
@Description: 
'''

import random
import cjjpy as cjj
import sys, os
import torch
from tqdm import tqdm

try:
    from .t5_qg.generator import Generator
except:
    sys.path.append(cjj.AbsParentDir(__file__, '.'))
    from t5_qg.generator import Generator


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


class QuestionGenerator:
    def __init__(self, model, prefix=None, verbose=True):
        assert model in ['t5']
        self.verbose = verbose
        prefix = f'{prefix}/models/question_generation/t5-base-qg-hl/' if prefix else None
        self.qg = Generator('valhalla/t5-base-qg-hl', prefix,
                            device='cuda' if torch.cuda.is_available() else 'cpu',
                            verbose=self.verbose)

    def _clean_input_lines(self, input_lines):
        # Only use the first option
        if isinstance(input_lines[0][1], tuple) and len(input_lines[0][1]) == 3:
            input_lines = list(map(lambda x: (x[0], [x[1]]), input_lines))
        return input_lines

    def generate(self, input_lines: list, sample_num=1, batch_size=128, mask_token='<mask>'):
        '''
        :param input_lines: List([text, options=[('answer', 0, 1), (x, y, z), ...]])
        :param sample_num: default as 1, as usually only provide one option.
        :return: List((regular_q, cloze_q, a))
        '''
        qa_pairs = []
        if len(input_lines) == 0:
            return qa_pairs
        input_lines = self._clean_input_lines(input_lines)
        ques_chunk = []

        for text, options in input_lines:
            masked_qa = self.mask_text(text, options, sample_num=sample_num, mask_token=mask_token)
            for q, a in masked_qa:
                ques_chunk.append({'context': text, 'answer': a, 'cloze_q': q})

        ques_pairs = self.qg(ques_chunk, batch_size=batch_size)
        iter = tqdm(zip(ques_pairs, ques_chunk), desc='Replacing') \
            if self.verbose else zip(ques_pairs, ques_chunk)
        for qa, mq in iter:
            q = qa['questions'][0]
            a = qa['answer']
            q = q.replace(a[0], mask_token)
            qa_pairs.append((q, mq['cloze_q'], a))

        return qa_pairs

    def _sample(self, options, sample_num=1):
        if len(options) <= sample_num:
            return options
        else:
            return random.sample(options, sample_num)

    def mask_text(self, text: str, options, sample_num=1, mask_token='<mask>'):
        '''
        :param text: text
        :param options: [('xx', 1, 2), (), ()]
        :return: [text, ('xx', 1, 2)] * sample_num
        '''
        masked_span = self._sample(options, sample_num)
        masked = []
        for span in masked_span:
            if isinstance(span, str):
                ntext = text.replace(span, mask_token)
            elif len(span) == 3:
                assert text[span[1]:span[2]] == span[0], (text[span[1]:span[2]], span[0])
                ntext = text[:span[1]] + mask_token + text[span[2]:]
            else:
                raise ValueError(span)
            masked.append((ntext, span))
        return masked

    def assemble_question(self, regular_q, cloze_q):
        return f'{regular_q} or {cloze_q}'


if __name__ == '__main__':
    qg = QuestionGenerator('t5')
    qa_pairs = qg.generate([['I was born yesterday.', [('born', 6, 10), ('yesterday', 11, 20)]]], sample_num=1)
    print(qa_pairs)
