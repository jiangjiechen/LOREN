# -*- coding: utf-8 -*-

"""
@Author             : Bao
@Date               : 2020/4/14
@Desc               :
@Last modified by   : Bao
@Last modified date : 2020/8/12
"""

import os
import copy
import logging
import ujson as json
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset
import tensorflow as tf
import cjjpy as cjj
import sys

try:
    from ...mrc_client.answer_generator import assemble_answers_to_one
except:
    sys.path.append(cjj.AbsParentDir(__file__, '...'))
    from mrc_client.answer_generator import assemble_answers_to_one

logger = logging.getLogger(__name__)


class InputExample(object):
    def __init__(self, guid, claim, evidences, questions, answers,
                 evidential, label=None, nli_labels=None):
        self.guid = guid
        self.claim = claim
        self.evidences = evidences
        self.questions = questions
        self.answers = answers
        self.evidential = evidential
        self.label = label
        self.nli_labels = nli_labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    def __init__(
            self,
            guid,
            c_input_ids,
            c_attention_mask,
            c_token_type_ids,
            q_input_ids_list,
            q_attention_mask_list,
            q_token_type_ids_list,
            nli_labels=None,
            label=None,
    ):
        self.guid = guid
        self.c_input_ids = c_input_ids
        self.c_attention_mask = c_attention_mask
        self.c_token_type_ids = c_token_type_ids
        self.q_input_ids_list = q_input_ids_list
        self.q_attention_mask_list = q_attention_mask_list
        self.q_token_type_ids_list = q_token_type_ids_list
        self.nli_labels = nli_labels
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def _create_input_ids_from_token_ids(token_ids_a, token_ids_b, tokenizer, max_seq_length):
    pair = len(token_ids_b) != 0

    # Truncate sequences.
    num_special_tokens_to_add = tokenizer.num_special_tokens_to_add(pair=pair)
    while len(token_ids_a) + len(token_ids_b) > max_seq_length - num_special_tokens_to_add:
        if len(token_ids_b) > 0:
            token_ids_b = token_ids_b[:-1]
        else:
            token_ids_a = token_ids_a[:-1]

    # Add special tokens to input_ids.
    input_ids = tokenizer.build_inputs_with_special_tokens(token_ids_a, token_ids_b if pair else None)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
    attention_mask = [1] * len(input_ids)

    # Create token_type_ids.
    token_type_ids = tokenizer.create_token_type_ids_from_sequences(token_ids_a, token_ids_b if pair else None)

    # Pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)
    if tokenizer.padding_side == "right":
        input_ids = input_ids + ([tokenizer.pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([tokenizer.pad_token_type_id] * padding_length)
    else:
        input_ids = ([tokenizer.pad_token_id] * padding_length) + input_ids
        attention_mask = ([0] * padding_length) + attention_mask
        token_type_ids = ([tokenizer.pad_token_type_id] * padding_length) + token_type_ids

    assert len(input_ids) == max_seq_length
    assert len(attention_mask) == max_seq_length
    assert len(token_type_ids) == max_seq_length

    return input_ids, attention_mask, token_type_ids


def convert_examples_to_features(
        examples,
        tokenizer,
        max_seq1_length=256,
        max_seq2_length=128,
        verbose=True
):
    features = []
    iter = tqdm(examples, desc="Converting Examples") if verbose else examples
    for (ex_index, example) in enumerate(iter):
        encoded_outputs = {"guid": example.guid, 'label': example.label,
                           'nli_labels': example.nli_labels}

        # ****** for sequence 1 ******* #
        token_ids_a, token_ids_b = [], []

        # text a in sequence 1
        token_ids = tokenizer.encode(example.claim, add_special_tokens=False)  # encode claim
        token_ids_a.extend(token_ids)

        # text b in sequence 1
        for i, evidence in enumerate(example.evidences):
            token_ids = tokenizer.encode(evidence, add_special_tokens=False)  # encode evidence
            token_ids_b.extend(token_ids + [tokenizer.sep_token_id])
        # Remove last sep token in token_ids_b.
        token_ids_b = token_ids_b[:-1]
        token_ids_b = token_ids_b[:max_seq1_length - len(token_ids_a) - 4]  # magic number for special tokens

        # premise </s> </s> hypothesis
        input_ids, attention_mask, token_type_ids = _create_input_ids_from_token_ids(
            token_ids_b,
            token_ids_a,
            tokenizer,
            max_seq1_length,
        )

        encoded_outputs["c_input_ids"] = input_ids
        encoded_outputs["c_attention_mask"] = attention_mask
        encoded_outputs["c_token_type_ids"] = token_type_ids

        # ****** for sequence 2 ******* #
        encoded_outputs["q_input_ids_list"] = []  # m x L
        encoded_outputs["q_attention_mask_list"] = []
        encoded_outputs["q_token_type_ids_list"] = []

        for candidate in example.evidential:
            # text a in sequence 2
            token_ids_a = tokenizer.encode(example.claim, add_special_tokens=False)  # encode claim
            # text b in sequence 2
            token_ids_b = tokenizer.encode(candidate, add_special_tokens=False)  # encode candidate answer
            # premise </s> </s> hypothesis
            input_ids, attention_mask, token_type_ids = _create_input_ids_from_token_ids(
                token_ids_b,
                token_ids_a,
                tokenizer,
                max_seq2_length,
            )

            encoded_outputs["q_input_ids_list"].append(input_ids)
            encoded_outputs["q_attention_mask_list"].append(attention_mask)
            encoded_outputs["q_token_type_ids_list"].append(token_type_ids)

        features.append(InputFeatures(**encoded_outputs))

        if ex_index < 5 and verbose:
            logger.info("*** Example ***")
            logger.info("guid: {}".format(example.guid))
            logger.info("c_input_ids: {}".format(encoded_outputs["c_input_ids"]))
            for input_ids in encoded_outputs['q_input_ids_list']:
                logger.info('q_input_ids: {}'.format(input_ids))
            logger.info("label: {}".format(example.label))
            logger.info("nli_labels: {}".format(example.nli_labels))

    return features


class DataProcessor:
    def __init__(
            self,
            model_name_or_path,
            max_seq1_length,
            max_seq2_length,
            max_num_questions,
            cand_k,
            data_dir='',
            cache_dir_name='cache_check',
            overwrite_cache=False,
            mask_rate=0.
    ):
        self.model_name_or_path = model_name_or_path
        self.max_seq1_length = max_seq1_length
        self.max_seq2_length = max_seq2_length
        self.max_num_questions = max_num_questions
        self.k = cand_k
        self.mask_rate = mask_rate

        self.data_dir = data_dir
        self.cached_data_dir = os.path.join(data_dir, cache_dir_name)
        self.overwrite_cache = overwrite_cache

        self.label2id = {"SUPPORTS": 2, "REFUTES": 0, 'NOT ENOUGH INFO': 1}

    def _format_file(self, role):
        return os.path.join(self.data_dir, "{}.json".format(role))

    def load_and_cache_data(self, role, tokenizer, data_tag):
        tf.io.gfile.makedirs(self.cached_data_dir)
        cached_file = os.path.join(
            self.cached_data_dir,
            "cached_features_{}_{}_{}_{}_{}_{}".format(
                role,
                list(filter(None, self.model_name_or_path.split("/"))).pop(),
                str(self.max_seq1_length),
                str(self.max_seq2_length),
                str(self.k),
                data_tag
            ),
        )
        if os.path.exists(cached_file) and not self.overwrite_cache:
            logger.info("Loading features from cached file {}".format(cached_file))
            features = torch.load(cached_file)
        else:
            examples = []
            with tf.io.gfile.GFile(self._format_file(role)) as f:
                data = f.readlines()
                for line in tqdm(data):
                    sample = self._load_line(line)
                    examples.append(InputExample(**sample))
            features = convert_examples_to_features(examples, tokenizer,
                                                    self.max_seq1_length, self.max_seq2_length)
            if 'train' in role or 'eval' in role:
                logger.info("Saving features into cached file {}".format(cached_file))
                torch.save(features, cached_file)

        return self._create_tensor_dataset(features, tokenizer)

    def convert_inputs_to_dataset(self, inputs, tokenizer, verbose=True):
        examples = []
        for line in inputs:
            sample = self._load_line(line)
            examples.append(InputExample(**sample))
        features = convert_examples_to_features(examples, tokenizer,
                                                self.max_seq1_length, self.max_seq2_length, verbose)

        return self._create_tensor_dataset(features, tokenizer, do_predict=True)

    def _create_tensor_dataset(self, features, tokenizer, do_predict=False):
        all_c_input_ids = torch.tensor([f.c_input_ids for f in features], dtype=torch.long)
        all_c_attention_mask = torch.tensor([f.c_attention_mask for f in features], dtype=torch.long)
        all_c_token_type_ids = torch.tensor([f.c_token_type_ids for f in features], dtype=torch.long)

        all_q_input_ids_list = []
        all_q_attention_mask_list = []
        all_q_token_type_ids_list = []

        def _trunc_agg(self, feature, pad_token):
            # feature: m x L
            _input_list = [v for v in feature[:self.max_num_questions]]
            while len(_input_list) < self.max_num_questions:
                _input_list.append([pad_token] * self.max_seq2_length)
            return _input_list

        for f in features:  # N x m x L
            all_q_input_ids_list.append(_trunc_agg(self, f.q_input_ids_list, tokenizer.pad_token_id))
            all_q_attention_mask_list.append(_trunc_agg(self, f.q_attention_mask_list, 0))
            all_q_token_type_ids_list.append(_trunc_agg(self, f.q_token_type_ids_list, tokenizer.pad_token_type_id))

        all_q_input_ids_list = torch.tensor(all_q_input_ids_list, dtype=torch.long)
        all_q_attention_mask_list = torch.tensor(all_q_attention_mask_list, dtype=torch.long)
        all_q_token_type_ids_list = torch.tensor(all_q_token_type_ids_list, dtype=torch.long)

        all_nli_labels_list = []
        for f in features:
            all_nli_labels_list.append(f.nli_labels[:self.max_num_questions]
                                       + max(0, (self.max_num_questions - len(f.nli_labels))) * [[0., 0., 0.]])
        all_nli_labels = torch.tensor(all_nli_labels_list, dtype=torch.float)

        if not do_predict:
            all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
            dataset = TensorDataset(
                all_c_input_ids, all_c_attention_mask, all_c_token_type_ids,
                all_q_input_ids_list, all_q_attention_mask_list, all_q_token_type_ids_list,
                all_nli_labels, all_labels,
            )
        else:
            dataset = TensorDataset(
                all_c_input_ids, all_c_attention_mask, all_c_token_type_ids,
                all_q_input_ids_list, all_q_attention_mask_list, all_q_token_type_ids_list,
                all_nli_labels,
            )

        return dataset

    def _load_line(self, line):
        if isinstance(line, str):
            line = json.loads(line)
        guid = line["id"]
        claim = line["claim"]

        # TODO: hack no evidence situation
        evidences = line["evidence"] if len(line['evidence']) > 0 else ['no idea'] * 5
        questions = line["questions"]
        answers = line["answers"]
        evidential = assemble_answers_to_one(line, self.k, mask_rate=self.mask_rate)['evidential_assembled']
        label = line.get("label", None)
        nli_labels = line.get('nli_labels', [[0., 0., 0.]] * len(questions))

        for i, e in enumerate(evidential):
            if '<mask>' in e:
                nli_labels[i] = [0., 0., 0.]

        answers = [v[0] for v in answers]  # k = 1
        label = self.label2id.get(label)

        sample = {
            "guid": guid,
            "claim": claim,
            "evidences": evidences,
            "questions": questions,
            "answers": answers,
            "evidential": evidential,  # already assembled.
            "label": label,
            'nli_labels': nli_labels
        }
        return sample
