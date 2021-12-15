# -*- coding: utf-8 -*-

'''
@Author     : Jiangjie Chen
@Time       : 2020/5/11 19:08
@Contact    : jjchen19@fudan.edu.cn
@Description: 
'''

import os
import tagme


def read_title_id(entity_def_path):
    id_to_title = {}
    with open(entity_def_path, 'r', encoding='UTF-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i > 0:
                entity, id = line.strip().split('|')
                id_to_title[id] = entity

    return id_to_title


class ELClient:
    def __init__(self, link_type, min_rho=0.1, prefix=None, verbose=False):
        self.verbose = verbose
        self.link_type = link_type
        if link_type == 'tagme':
            self.min_rho = min_rho
            tagme.GCUBE_TOKEN = os.environ['TAGME_APIKEY']
        elif link_type == 'spacy':
            assert prefix is not None
            self.init_spacy_linker(prefix)
        else:
            raise NotImplementedError(link_type)

    def init_spacy_linker(self, prefix):
        entity_def_path = f"{prefix}/entity_defs.csv"
        self._print('* Loading entity linker...')
        self.nlp = spacy.load(prefix)
        self.id2title = read_title_id(entity_def_path)
        self._print('* Entity linker loaded.')

    def _tagme_link(self, text):
        result = []
        for ann in tagme.annotate(text, long_text=1).get_annotations(min_rho=self.min_rho):
            result.append((text[ann.begin:ann.end], ann.score, ann.entity_id, ann.entity_title))
            # result.append({'begin': ann.begin,
            #                'end': ann.end,
            #                'id': ann.entity_id,
            #                'title': ann.entity_title,
            #                'score': ann.score})
        result.sort(key=lambda x: x[1], reverse=True)
        return result

    def link(self, text):
        if self.link_type == 'tagme':
            return self._tagme_link(text)
        else:
            return self._spacy_link(text)

    def _spacy_link(self, text):
        text = self._preprocess_text(text)
        doc = self.nlp(text)
        ents = [(e.text, e.label_, e.kb_id_, self.id2title.get(e.kb_id_, ''))
                for e in doc.ents if e.kb_id_ != 'NIL']
        return ents

    def _preprocess_text(self, text):
        if isinstance(text, list):
            text = ' '.join(text)
        text = text.strip().replace('-lrb-', '(').replace('-rrb-', ')')
        return text

    def _print(self, x):
        if self.verbose: print(x)


if __name__ == '__main__':
    elcl = ELClient(link_type='tagme', verbose=True)
    res = elcl.link('Jeff Dean wants to meet Yoshua Bengio.')
    print(res)
