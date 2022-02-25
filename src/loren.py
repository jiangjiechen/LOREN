# -*- coding: utf-8 -*-

'''
@Author     : Jiangjie Chen
@Time       : 2020/9/17 15:55
@Contact    : jjchen19@fudan.edu.cn
@Description: 
'''

import os
import sys
import json
import logging
import cjjpy as cjj

try:
    from .qg_client.question_generator import QuestionGenerator
    from .mrc_client.answer_generator import AnswerGenerator, chunks, assemble_answers_to_one
    from .parsing_client.sentence_parser import SentenceParser, deal_bracket
    from .check_client.fact_checker import FactChecker, id2label
    from .er_client import EvidenceRetrieval
except:
    sys.path.append(cjj.AbsParentDir(__file__, '.'))
    from qg_client.question_generator import QuestionGenerator
    from mrc_client.answer_generator import AnswerGenerator, chunks, assemble_answers_to_one
    from parsing_client.sentence_parser import SentenceParser, deal_bracket
    from check_client.fact_checker import FactChecker, id2label
    from er_client import EvidenceRetrieval


def load_config(config):
    if isinstance(config, str):
        with open(config) as f:
            config = json.load(f)
    cfg = cjj.AttrDict(config)
    return cfg


class Loren:
    def __init__(self, config_file, verbose=True):
        self.verbose = verbose
        self.args = load_config(config_file)
        self.sent_client = SentenceParser()
        self.qg_client = QuestionGenerator('t5', verbose=False)
        self.ag_client = AnswerGenerator(self.args.mrc_dir)
        self.fc_client = FactChecker(self.args, self.args.fc_dir)
        self.er_client = EvidenceRetrieval(self.args.er_dir)
        self.logger = cjj.init_logger(f'{os.environ["PJ_HOME"]}/results/loren_dev.log',
                                      log_file_level=logging.INFO if self.verbose else logging.WARNING)
        self.logger.info('*** Loren initialized. ***')

    def check(self, claim, evidence=None):
        self.logger.info('*** Verifying "%s"... ***' % claim)
        js = self.prep(claim, evidence)
        js['id'] = 0
        y_predicted, z_predicted, m_attn = self.fc_client.check_from_batch([js], verbose=self.verbose)
        label = id2label[y_predicted[0]]

        # Update js
        js = assemble_answers_to_one(js, k=self.args.cand_k)
        js['evidence'] = [self.fc_client.tokenizer.clean_up_tokenization(e) for e in js['evidence']]
        js['questions'] = [self.fc_client.tokenizer.clean_up_tokenization(q) for q in js['questions']]
        js['claim_phrases'] = [self.fc_client.tokenizer.clean_up_tokenization(a[0]) for a in js['answers']]
        js['local_premises'] = [self.fc_client.tokenizer.clean_up_tokenization(a) for a in js['evidential_assembled']]
        # js['m_attn'] = m_attn[0][:len(js['claim_phrases'])]
        js['phrase_veracity'] = z_predicted[0][:len(js['claim_phrases'])]
        js['claim_veracity'] = label

        self.logger.info("  * Intermediary: %s *" % str(js))
        self.logger.info('*** Verification completed: "%s" ***' % label)
        return js

    def prep(self, claim, evidence=None):
        '''
        :param evidence: 'aaa||bbb||ccc' / [entity, num, evidence, (prob)] if not None
        '''
        evidence_tuples = self._prep_evidence(claim, evidence)
        evidence = [x[2] for x in evidence_tuples]
        entity = [x[0] for x in evidence_tuples]
        self.logger.info('  * Evidence prepared. *')
        assert isinstance(evidence, list)

        js = {'claim': claim, 'evidence': evidence, 'entities': entity}
        js = self._prep_claim_phrases(js)
        self.logger.info('  * Claim phrases prepared. *')
        js = self._prep_questions(js)
        self.logger.info('  * Probing questions prepared. *')
        js = self._prep_evidential_phrases(js)
        self.logger.info('  * Evidential phrases prepared. *')
        return js

    def _prep_claim_phrases(self, js):
        results = self.sent_client.identify_NPs(deal_bracket(js['claim'], True),
                                                candidate_NPs=[x[0] for x in js['evidence']])
        NPs = results['NPs']
        claim_tokenized = results['text']
        verbs = results['verbs']
        adjs = results['adjs']
        js['claim_tokenized'] = claim_tokenized
        js['answers'] = NPs + verbs + adjs
        js['answer_roles'] = ['noun'] * len(NPs) + ['verb'] * len(verbs) + ['adj'] * len(adjs)
        if len(js['answers']) == 0:
            js['answers'] = js['claim'].split()[0]
            js['answer_roles'] = ['noun']
        return js

    def _prep_questions(self, js):
        _cache = []
        for answer in js['answers']:
            _cache.append((js['claim_tokenized'], [answer]))
        qa_pairs = self.qg_client.generate([(x, y) for x, y in _cache])
        for q, clz_q, a in qa_pairs:
            if 'questions' in js:
                js['regular_qs'].append(q)
                js['cloze_qs'].append(clz_q)
                js['questions'].append(self.qg_client.assemble_question(q, clz_q))
            else:
                js['regular_qs'] = [q]
                js['cloze_qs'] = [clz_q]
                js['questions'] = [self.qg_client.assemble_question(q, clz_q)]
        return js

    def _prep_evidential_phrases(self, js):
        examples = []
        for q in js['questions']:
            ex = self.ag_client.assemble(q, " ".join(js['evidence']))
            examples.append(ex)
        predicted = self.ag_client.generate(examples, num_beams=self.args['cand_k'],
                                            num_return_sequences=self.args['cand_k'],
                                            batch_size=2, verbose=False)
        for answers in predicted:
            if 'evidential' in js:
                js['evidential'].append(answers)
            else:
                js['evidential'] = [answers]
        return js

    def _prep_evidence(self, claim, evidence=None):
        '''
        :param evidence: 'aaa||bbb||ccc' / [entity, num, evidence, (prob)]
        :return: [entity, num, evidence, (prob)]
        '''
        if evidence in [None, '', 'null', 'NULL', 'Null']:
            evidence = self.er_client.retrieve(claim)
            evidence = [(ev[0], ev[1], deal_bracket(ev[2], True, ev[0])) for ev in evidence]
        else:
            if isinstance(evidence, str):
                # TODO: magic sentence number (5)
                evidence = [(None, None, ev.strip()) for i, ev in enumerate(evidence.split('||')[:5])]
        return evidence


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True,
                        default='available_models/aaai22_roberta.json',
                        help='Config json file with hyper-parameters')
    args = parser.parse_args()

    loren = Loren(args.config)
    while True:
        claim = input('> ')
        label, js = loren.check(claim)
        print(label)
        print(js)
