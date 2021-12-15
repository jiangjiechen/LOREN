# -*- coding:utf-8  -*-

"""
@Last modified date : 2020/12/23
"""

import re
import nltk
import torch.cuda
from nltk.stem import WordNetLemmatizer
from allennlp.predictors.predictor import Predictor

nltk.download('wordnet')
nltk.download('stopwords')


def deal_bracket(text, restore, leading_ent=None):
    if leading_ent:
        leading_ent = ' '.join(leading_ent.split('_'))
        text = f'Things about {leading_ent}: ' + text
    if restore:
        text = text.replace('-LRB-', '(').replace('-RRB-', ')')
        text = text.replace('LRB', '(').replace('RRB', ')')
    return text


def refine_entity(entity):
    entity = re.sub(r'-LRB- .+ -RRB-$', '', entity)
    entity = re.sub(r'LRB .+ RRB$', '', entity)
    entity = re.sub(r'_', ' ', entity)
    entity = re.sub(r'\s+', ' ', entity)
    return entity.strip()


def find_sub_seq(seq_a, seq_b, shift=0, uncased=False, lemmatizer=None):
    if uncased:
        seq_a = [token.lower() for token in seq_a]
        seq_b = [token.lower() for token in seq_b]
    if lemmatizer is not None:
        seq_a = [lemmatizer.lemmatize(token) for token in seq_a]
        seq_b = [lemmatizer.lemmatize(token) for token in seq_b]
    for i in range(shift, len(seq_a)):
        if seq_a[i:i+len(seq_b)] == seq_b:
            return i, i + len(seq_b)
    return -1, -1


def is_sub_seq(seq_start, seq_end, all_seqs):
    for start, end, is_candidate in all_seqs:
        if start <= seq_start < seq_end <= end:
            return start, end, is_candidate
    return None


# extract named entity with B-I-L-U-O schema
def extract_named_entity(tags):
    all_NEs = []
    ne_type, ne_start = '', -1
    for i, t in enumerate(tags):
        if t == 'O':
            ne_type, ne_start = '', -1
            continue
        t1, t2 = t.split('-')
        if t1 == 'B':
            ne_type, ne_start = t2, i
        elif t1 == 'I' and t2 != ne_type:
            ne_type, ne_start = '', -1
        elif t1 == 'L' and t2 != ne_type:
            ne_type, ne_start = '', -1
        elif t1 == 'L' and t2 == ne_type:
            all_NEs.append((ne_start, i + 1, False))
            ne_type, ne_start = '', -1
        elif t1 == 'U':
            all_NEs.append((i, i + 1, False))
            ne_type, ne_start = '', -1

    return all_NEs


def refine_results(tokens, spans, stopwords):
    all_spans = []
    for span_start, span_end, is_candidate in spans:
        # remove stopwords
        if not is_candidate:
            while span_start < span_end and tokens[span_start].lower() in stopwords:
                span_start += 1
            if span_start >= span_end:
                continue

        # add prefix
        if span_start > 0 and tokens[span_start - 1] in ['a', 'an', 'A', 'An', 'the', 'The']:
            span_start -= 1

        # convert token-level index into char-level index
        span = ' '.join(tokens[span_start:span_end])
        span_start = len(' '.join(tokens[:span_start])) + 1 * min(1, span_start)  # 1 for blank
        span_end = span_start + len(span)

        all_spans.append((span, span_start, span_end))
    all_spans = sorted(all_spans, key=lambda x: (x[1], x[1] - x[2]))

    # remove overlap
    refined_spans = []
    for span, span_start, span_end in all_spans:
        flag = True
        for _, start, end in refined_spans:
            if start <= span_start < span_end <= end:
                flag = False
                break
        if flag:
            refined_spans.append((span, span_start, span_end))

    return refined_spans


class SentenceParser:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu',
                 ner_path="https://storage.googleapis.com/allennlp-public-models/ner-model-2020.02.10.tar.gz",
                 cp_path="https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz"):
        self.device = self.parse_device(device)
        self.ner = Predictor.from_path(ner_path, cuda_device=self.device)
        print('* ner loaded')
        self.cp = Predictor.from_path(cp_path, cuda_device=self.device)
        print('* constituency parser loaded')
        self.lemmatizer = WordNetLemmatizer()

        # some heuristic rules can be added here
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        self.stopwords.update({'-', '\'s', 'try', 'tries', 'tried', 'trying',
                               'become', 'becomes', 'became', 'becoming',
                               'make', 'makes', 'made', 'making', 'call', 'called', 'calling',
                               'put', 'ever', 'something', 'someone', 'sometime'})
        self.special_tokens = ['only', 'most', 'before', 'after', 'behind']
        for token in self.special_tokens:
            if token in self.stopwords: self.stopwords.remove(token)
        if 'won' in self.stopwords: self.stopwords.remove('won')
        if 'own' in self.stopwords: self.stopwords.remove('own')

    def parse_device(self, device):
        if 'cpu' in device:
            return -1
        else:
            dev = re.findall('\d+', device)
            return 0 if len(dev) == 0 else int(dev[0])

    def identify_NPs(self, text, candidate_NPs=None):
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text) == 0: return {'text': '', 'NPs': [], 'verbs': [], 'adjs': []}

        cp_outputs = self.cp.predict(text)
        ner_outputs = self.ner.predict(text)
        tokens = cp_outputs['tokens']
        pos_tags = cp_outputs['pos_tags']
        ner_tags = ner_outputs['tags']
        tree = cp_outputs['hierplane_tree']['root']

        # extract candidate noun phrases passed by user with token index
        all_NPs = []
        candidate_NPs = [refine_entity(np).split() for np in candidate_NPs] if candidate_NPs else []
        for np in sorted(candidate_NPs, key=len, reverse=True):
            np_start, np_end = find_sub_seq(tokens, np, 0, uncased=True, lemmatizer=self.lemmatizer)
            while np_start != -1 and np_end != -1:
                if not is_sub_seq(np_start, np_end, all_NPs):
                    all_NPs.append((np_start, np_end, True))
                np_start, np_end = find_sub_seq(tokens, np, np_end, uncased=True, lemmatizer=self.lemmatizer)

        # extract noun phrases from tree
        def _get_bottom_NPs(children):
            if 'children' not in children:
                return None
            if {'NP', 'OP', 'XP', 'QP'} & set(children['attributes']):
                is_bottom = True
                for child in children['children']:
                    if 'children' in child:
                        is_bottom = False
                if is_bottom:
                    bottom_NPs.append(children['word'].split())
                else:
                    for child in children['children']:
                        _get_bottom_NPs(child)
            else:
                for child in children['children']:
                    _get_bottom_NPs(child)
        bottom_NPs = []
        _get_bottom_NPs(tree)

        # find token indices of noun phrases
        np_index = -1
        for np in bottom_NPs:
            np_start, np_end = find_sub_seq(tokens, np, np_index + 1)
            if not is_sub_seq(np_start, np_end, all_NPs):
                all_NPs.append((np_start, np_end, False))
            np_index = np_end

        # extract named entities with token index
        all_NEs = extract_named_entity(ner_tags)

        # extract verbs with token index
        all_verbs = []
        for i, pos in enumerate(pos_tags):
            if pos[0] == 'V':
                if not is_sub_seq(i, i + 1, all_NPs) and not is_sub_seq(i, i + 1, all_NEs):
                    all_verbs.append((i, i + 1, False))

        # extract modifiers with token index
        all_modifiers = []
        for i, (token, pos) in enumerate(zip(tokens, pos_tags)):
            if pos in ['JJ', 'RB']:  # adj. and adv.
                if not is_sub_seq(i, i + 1, all_NPs) and not is_sub_seq(i, i + 1, all_NEs):
                    all_modifiers.append((i, i + 1, False))
            elif token in self.special_tokens:
                if not is_sub_seq(i, i + 1, all_NPs) and not is_sub_seq(i, i + 1, all_NEs):
                    all_modifiers.append((i, i + 1, False))

        # split noun phrases with named entities
        all_spans = []
        for np_start, np_end, np_is_candidate in all_NPs:
            if np_is_candidate:  # candidate noun phrases will be preserved
                all_spans.append((np_start, np_end, np_is_candidate))
            else:
                match = is_sub_seq(np_start, np_end, all_NEs)
                if match:  # if a noun phrase is a sub span of a named entity, the named entity will be preserved
                    all_spans.append(match)
                else:  # else if a named entity is a sub span of a noun phrase, the noun phrase will be split
                    index = np_start
                    for ne_start, ne_end, ne_is_candidate in all_NEs:
                        if np_start <= ne_start < ne_end <= np_end:
                            all_modifiers.append((index, ne_start, False))
                            all_spans.append((ne_start, ne_end, ne_is_candidate))
                            index = ne_end
                    all_spans.append((index, np_end, False))

        # named entities without overlapping
        for ne_start, ne_end, is_candidate in all_NEs:
            if not is_sub_seq(ne_start, ne_end, all_spans):
                all_spans.append((ne_start, ne_end, is_candidate))

        all_spans = refine_results(tokens, all_spans, self.stopwords)
        all_verbs = refine_results(tokens, all_verbs, self.stopwords)
        all_modifiers = refine_results(tokens, all_modifiers, self.stopwords)

        return {'text': tree['word'], 'NPs': all_spans, 'verbs': all_verbs, 'adjs': all_modifiers}


if __name__ == '__main__':
    import json

    print('Initializing sentence parser.')
    client = SentenceParser(device='cpu')

    print('Parsing sentence.')
    sentence = "The Africa Cup of Nations is held in odd - numbered years due to conflict with the World Cup . "
    entities = ['Africa Cup of Nations', 'Africa_Cup_of_Nations', 'Africa Cup', 'Africa_Cup']
    results = client.identify_NPs(sentence, entities)
    print(json.dumps(results, ensure_ascii=False, indent=4))

    # import random
    # from tqdm import tqdm
    # from utils import read_json_lines, save_json
    #
    # print('Parsing file.')
    # results = []
    # data = list(read_json_lines('data/train.jsonl'))
    # random.shuffle(data)
    # for entry in tqdm(data[:100]):
    #     results.append(client.identify_NPs(entry['claim']))
    # save_json(results, 'data/results.json')
