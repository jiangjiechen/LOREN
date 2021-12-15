# -*- coding: utf-8 -*-

'''
@Author     : Jiangjie Chen
@Time       : 2020/11/12 21:19
@Contact    : jjchen19@fudan.edu.cn
@Description: 
'''

import wikipediaapi
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
try:
    from entitylinker import ELClient
except:
    from .entitylinker import ELClient


class DocRetrieval:
    def __init__(self, link_type):
        self.wiki = wikipediaapi.Wikipedia('en')
        self.er_client = ELClient(link_type, verbose=True)

    def _get_page(self, title):
        summary = self.wiki.page(title).summary
        sents = []
        for i, sent in enumerate(sent_tokenize(summary)):
            sents.append((title, i, sent, 0))
        return sents

    def retrieve_docs(self, claim):
        el_results = self.er_client.link(claim)
        sents = []
        for text, label, kb_id, title in el_results:
            if title == '': continue
            sents += self._get_page(title)
        return sents


if __name__ == '__main__':
    doc = DocRetrieval('tagme')
    print(doc.retrieve_docs('joe biden won the U.S. president.'))
    print(doc.retrieve_docs('Joe Biden won the U.S. president.'))