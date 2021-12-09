# -*- coding: utf-8 -*-

'''
@Author     : Jiangjie Chen
@Time       : 2019/9/17 13:58
@Contact    : mi0134sher@hotmail.com
@Description: 
'''

import spacy
import os

nlp = None

class BasicNLP:
	def __init__(self, s, lang='en', processors='tokenize,pos,lemma,depparse,ner'):
		'''
		:param s: sentence string
		:param processors: 'tokenize,pos,lemma,depparse,ner'
		'''
		global nlp

		self.lang2pak = {"en": "en_core_web_sm", "zh": "zh_core_web_sm"}
		assert lang in self.lang2pak, "Only support {}".format(list(self.lang2pak.keys()))
		self._load_nlp(lang)

		self.words, self.depparse, self.postags, self.lemma, self.ner = [], [], [], [], []
		if len(s.strip()) > 0:
			self._doc = nlp(s)
			for sent in self._doc.sents:
				if 'tokenize' in processors:
					self.words.append([w.text for w in sent])
				if 'lemma' in processors:
					self.lemma.append([w.lemma_ for w in sent])
				if 'pos' in processors:
					self.postags.append([w.pos_ for w in sent])
				if 'depparse' in processors:
					self.depparse.append([(w.dep_, w.head.i, i) for i, w in enumerate(sent)])
				if 'ner' in processors:
					self.ner.append(sent.ents)

	def show_all(self):
		print('* words, lemma, postags')
		print(self.words)
		print(self.lemma)
		print(self.postags)
		print('* dependency parsing')
		print(self.depparse)
		print('* ner')
		print(self.ner)

	def _load_nlp(self, lang):
		global nlp
		if nlp is None or nlp.lang != lang:
			nlp_pak = self.lang2pak[lang]
			try:
				nlp = spacy.load(nlp_pak)
			except:
				os.system(f'python3 -m spacy download {nlp_pak}')
				nlp = spacy.load(nlp_pak)


if __name__ == '__main__':
	# b = Sentence("英国首相约翰逊6日晚因病情恶化，被转入重症监护室治疗。英国首相府发言人说，目前约翰逊意识清晰，将他转移到重症监护室只是预防性措施。", lang="zh")
	# b.show_all()

	a = BasicNLP("One True Thing is an American film.", lang="en")
	a.show_all()