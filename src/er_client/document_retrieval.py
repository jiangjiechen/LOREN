# -*- coding:utf-8  -*-

"""
@Author             : Bao
@Date               : 2020/9/17
@Desc               : Document selection and sentence ranking code from KGAT. Not used in LOREN.
@Last modified by   : Bao
@Last modified date : 2020/9/17
"""

import re
import time
import json
import nltk
from tqdm import tqdm
from allennlp.predictors import Predictor
from drqa.retriever import DocDB, utils
from drqa.retriever.utils import normalize
import wikipedia


class FeverDocDB(DocDB):
    def __init__(self, path=None):
        super().__init__(path)

    def get_doc_lines(self, doc_id):
        """Fetch the raw text of the doc for 'doc_id'."""
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT lines FROM documents WHERE id = ?",
            (utils.normalize(doc_id),)
        )
        result = cursor.fetchone()
        cursor.close()

        result = result[0] if result is not None else ''
        doc_lines = []
        for line in result.split('\n'):
            if len(line) == 0: continue
            line = line.split('\t')[1]
            if len(line) == 0: continue
            doc_lines.append((doc_id, len(doc_lines), line, 0))

        return doc_lines

    def get_non_empty_doc_ids(self):
        """Fetch all ids of docs stored in the db."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT id FROM documents WHERE length(trim(text)) > 0")
        results = [r[0] for r in cursor.fetchall()]
        cursor.close()
        return results


class DocRetrieval:
    def __init__(self, database_path, add_claim=False, k_wiki_results=None):
        self.db = FeverDocDB(database_path)
        self.add_claim = add_claim
        self.k_wiki_results = k_wiki_results
        self.porter_stemmer = nltk.PorterStemmer()
        self.tokenizer = nltk.word_tokenize
        self.predictor = Predictor.from_path(
            "https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz"
        )

    def get_NP(self, tree, nps):
        if isinstance(tree, dict):
            if "children" not in tree:
                if tree['nodeType'] == "NP":
                    # print(tree['word'])
                    # print(tree)
                    nps.append(tree['word'])
            elif "children" in tree:
                if tree['nodeType'] == "NP":
                    # print(tree['word'])
                    nps.append(tree['word'])
                    self.get_NP(tree['children'], nps)
                else:
                    self.get_NP(tree['children'], nps)
        elif isinstance(tree, list):
            for sub_tree in tree:
                self.get_NP(sub_tree, nps)

        return nps

    def get_subjects(self, tree):
        subject_words = []
        subjects = []
        for subtree in tree['children']:
            if subtree['nodeType'] == "VP" or subtree['nodeType'] == 'S' or subtree['nodeType'] == 'VBZ':
                subjects.append(' '.join(subject_words))
                subject_words.append(subtree['word'])
            else:
                subject_words.append(subtree['word'])
        return subjects

    def get_noun_phrases(self, claim):
        tokens = self.predictor.predict(claim)
        nps = []
        tree = tokens['hierplane_tree']['root']
        noun_phrases = self.get_NP(tree, nps)
        subjects = self.get_subjects(tree)
        for subject in subjects:
            if len(subject) > 0:
                noun_phrases.append(subject)
        if self.add_claim:
            noun_phrases.append(claim)
        return list(set(noun_phrases))

    def get_doc_for_claim(self, noun_phrases):
        predicted_pages = []
        for np in noun_phrases:
            if len(np) > 300:
                continue
            i = 1
            while i < 12:
                try:
                    # print(np)
                    # res = server.lookup(np, keep_all=True)
                    # docs = [y for _, y in res] if res is not None else []
                    docs = wikipedia.search(np)
                    if self.k_wiki_results is not None:
                        predicted_pages.extend(docs[:self.k_wiki_results])
                    else:
                        predicted_pages.extend(docs)
                except (ConnectionResetError, ConnectionError, ConnectionAbortedError, ConnectionRefusedError):
                    print("Connection reset error received! Trial #" + str(i))
                    time.sleep(600 * i)
                    i += 1
                else:
                    break

            # sleep_num = random.uniform(0.1,0.7)
            # time.sleep(sleep_num)
        predicted_pages = set(predicted_pages)
        processed_pages = []
        for page in predicted_pages:
            page = page.replace(" ", "_")
            page = page.replace("(", "-LRB-")
            page = page.replace(")", "-RRB-")
            page = page.replace(":", "-COLON-")
            processed_pages.append(page)

        return processed_pages

    def np_conc(self, noun_phrases):
        noun_phrases = set(noun_phrases)
        predicted_pages = []
        for np in noun_phrases:
            page = np.replace('( ', '-LRB-')
            page = page.replace(' )', '-RRB-')
            page = page.replace(' - ', '-')
            page = page.replace(' :', '-COLON-')
            page = page.replace(' ,', ',')
            page = page.replace(" 's", "'s")
            page = page.replace(' ', '_')

            if len(page) < 1:
                continue
            doc_lines = self.db.get_doc_lines(page)
            if len(doc_lines) > 0:
                predicted_pages.append(page)
        return predicted_pages

    def exact_match(self, claim):
        noun_phrases = self.get_noun_phrases(claim)
        wiki_results = self.get_doc_for_claim(noun_phrases)
        wiki_results = list(set(wiki_results))

        claim = claim.replace(".", "")
        claim = claim.replace("-", " ")
        words = [self.porter_stemmer.stem(word.lower()) for word in self.tokenizer(claim)]
        words = set(words)
        predicted_pages = self.np_conc(noun_phrases)

        for page in wiki_results:
            page = normalize(page)
            processed_page = re.sub("-LRB-.*?-RRB-", "", page)
            processed_page = re.sub("_", " ", processed_page)
            processed_page = re.sub("-COLON-", ":", processed_page)
            processed_page = processed_page.replace("-", " ")
            processed_page = processed_page.replace("–", " ")
            processed_page = processed_page.replace(".", "")
            page_words = [self.porter_stemmer.stem(word.lower()) for word in self.tokenizer(processed_page) if
                          len(word) > 0]

            if all([item in words for item in page_words]):
                if ':' in page:
                    page = page.replace(":", "-COLON-")
                predicted_pages.append(page)
        predicted_pages = list(set(predicted_pages))

        return noun_phrases, wiki_results, predicted_pages


def save_to_file(results, client, filename):
    with open(filename, 'w', encoding='utf-8') as fout:
        for _id, line in enumerate(results):
            claim = line['claim']
            evidence = []
            for page in line['predicted_pages']:
                evidence.extend(client.db.get_doc_lines(page))
            print(json.dumps({'claim': claim, 'evidence': evidence}, ensure_ascii=False), file=fout)


if __name__ == '__main__':
    database_path = 'data/fever.db'
    add_claim = True
    k_wiki_results = 7
    client = DocRetrieval(database_path, add_claim, k_wiki_results)

    results = []
    with open('data/claims.json', 'r', encoding='utf-8') as fin:
        for line in tqdm(fin):
            line = json.loads(line)
            _, _, predicted_pages = client.exact_match(line['claim'])
            evidence = []
            for page in predicted_pages:
                evidence.extend(client.db.get_doc_lines(page))
            line['evidence'] = evidence
            results.append(line)

    with open('data/pages.json', 'w', encoding='utf-8') as fout:
        for line in results:
            print(json.dumps(line, ensure_ascii=False), file=fout)
