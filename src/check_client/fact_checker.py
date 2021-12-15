# -*- coding: utf-8 -*-

"""
@Author             : Bao
@Date               : 2020/8/12
@Desc               :
@Last modified by   : Bao
@Last modified date : 2020/8/20
"""

import os
import sys
import logging
import torch
from tqdm import tqdm
import tensorflow as tf
import ujson as json
import argparse
import cjjpy as cjj
from itertools import repeat
from torch.utils.data import DataLoader, SequentialSampler
from transformers import (
    BertConfig, BertTokenizer, AutoTokenizer,
    RobertaConfig, RobertaTokenizer,
)

try:
    from .modules.data_processor import DataProcessor
    from .plm_checkers import BertChecker, RobertaChecker
    from .utils import read_json_lines, compute_metrics, set_seed
    from .train import do_evaluate
    from ..eval_client.fever_scorer import FeverScorer
except:
    sys.path.append(cjj.AbsParentDir(__file__, '.'))
    sys.path.append(cjj.AbsParentDir(__file__, '..'))
    from eval_client.fever_scorer import FeverScorer
    from modules.data_processor import DataProcessor
    from plm_checkers import BertChecker, RobertaChecker
    from utils import read_json_lines, compute_metrics, set_seed
    from train import do_evaluate

MODEL_MAPPING = {
    'bert': (BertConfig, BertTokenizer, BertChecker),
    'roberta': (RobertaConfig, RobertaTokenizer, RobertaChecker),
}

logger = logging.getLogger(__name__)
label2id = {"SUPPORTS": 2, "REFUTES": 0, 'NOT ENOUGH INFO': 1}
id2label = {v: k for k, v in label2id.items()}


class FactChecker:
    def __init__(self, args, fc_ckpt_dir=None, mask_rate=0.):
        self.data_processor = None
        self.tokenizer = None
        self.model = None
        self.args = args
        self.args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt = args.fc_dir if fc_ckpt_dir is None else fc_ckpt_dir
        self.mask_rate = mask_rate
        set_seed(args)
        logger.info('Initializing fact checker.')
        self._prepare_ckpt(self.args.model_name_or_path, self.ckpt)
        self.load_model()

    def _prepare_ckpt(self, model_name_or_path, ckpt_dir):
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        tokenizer.save_pretrained(ckpt_dir)

    def load_model(self):
        if self.model is None:
            self.data_processor = DataProcessor(
                self.args.model_name_or_path,
                self.args.max_seq1_length,
                self.args.max_seq2_length,
                self.args.max_num_questions,
                self.args.cand_k,
                mask_rate=self.mask_rate
            )

            _, tokenizer_class, model_class = MODEL_MAPPING[self.args.model_type]
            self.tokenizer = tokenizer_class.from_pretrained(
                self.ckpt,
                do_lower_case=self.args.do_lower_case
            )
            self.model = model_class.from_pretrained(
                self.ckpt,
                from_tf=bool(".ckpt" in self.ckpt),
                logic_lambda=self.args.logic_lambda,
                prior=self.args.prior,
            )
            if self.args.n_gpu > 0:
                self.model = torch.nn.DataParallel(self.model)

    def _check(self, inputs: list, batch_size=32, verbose=True):
        dataset = self.data_processor.convert_inputs_to_dataset(inputs, self.tokenizer, verbose=verbose)
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

        with torch.no_grad():
            self.model.to(self.args.device)
            self.model.eval()
            iter = tqdm(dataloader, desc="Fact Checking") if verbose else dataloader
            _, y_predicted, z_predicted, m_attn, mask = \
                do_evaluate(iter, self.model, self.args, during_training=False, with_label=False)

        return y_predicted, z_predicted, m_attn, mask

    def check_from_file(self, in_filename, out_filename, batch_size, verbose=False):
        if 'test' in in_filename:
            raw_inp = f'{os.environ["PJ_HOME"]}/data/fever/shared_task_test.jsonl'
        else:
            raw_inp = None
        tf.io.gfile.makedirs(os.path.dirname(out_filename))
        inputs = list(read_json_lines(in_filename))
        y_predicted, z_predicted, m_attn, mask = self._check(inputs, batch_size)

        z_predicted = repeat(None) if z_predicted is None else z_predicted
        m_attn = repeat(None) if m_attn is None else m_attn
        ordered_results = {}
        with_label = inputs[0].get('label') is not None

        if with_label:
            label_truth = [label2id[x['label']] for x in inputs]
            _, acc_results = compute_metrics(label_truth, y_predicted, z_predicted, mask)
        else:
            acc_results = {}

        for i, (inp, y, z, attn, _mask) in \
                enumerate(zip(inputs, y_predicted, z_predicted, m_attn, mask)):
            result = {'id': inp['id'],
                      'predicted_label': id2label[y],
                      'predicted_evidence': inp.get('predicted_evidence', [])}
            if verbose:
                if i < 5:
                    print("{}\t{}\t{}".format(inp.get("id", i), inp["claim"], y))
                if z is not None and attn is not None:
                    result.update({
                        'z_prob': z[:torch.tensor(_mask).sum()],
                        'm_attn': attn[:torch.tensor(_mask).sum()],
                    })
            ordered_results[inp['id']] = result

        with tf.io.gfile.GFile(out_filename, 'w') as fout:
            if raw_inp:
                with tf.io.gfile.GFile(raw_inp) as f:
                    for line in f:
                        raw_js = json.loads(line)
                        fout.write(json.dumps(ordered_results[raw_js['id']]) + '\n')
            else:
                for k in ordered_results:
                    fout.write(json.dumps(ordered_results[k]) + '\n')

        if ('dev' in in_filename or 'val' in in_filename) and with_label:
            scorer = FeverScorer()
            fever_results = scorer.get_scores(out_filename)
            fever_results.update(acc_results)

            print(fever_results)
            return fever_results

    def check_from_batch(self, inputs: list, verbose=False):
        y_predicted, z_predicted, m_attn, mask = self._check(inputs, len(inputs), verbose)
        return y_predicted, z_predicted, m_attn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True, type=str,
                        choices=['val', 'eval', 'test', 'demo'])
    parser.add_argument('--output', '-o', default='none', type=str)
    parser.add_argument('--ckpt', '-c', required=True, type=str)
    parser.add_argument('--model_type', default='roberta', type=str,
                        choices=['roberta', 'bert'])
    parser.add_argument('--model_name_or_path', default='roberta-large', type=str)
    parser.add_argument('--verbose', '-v', action='store_true', default=False,
                        help='whether output phrasal veracity or not')
    parser.add_argument('--logic_lambda', '-l', required=True, type=float)
    parser.add_argument('--prior', default='random', type=str, choices=['nli', 'uniform', 'logic', 'random'],
                        help='type of prior distribution')
    parser.add_argument('--mask_rate', '-m', default=0., type=float)

    parser.add_argument('--cand_k', '-k', default=3, type=int)
    parser.add_argument('--max_seq1_length', default=256, type=int)
    parser.add_argument('--max_seq2_length', default=128, type=int)
    parser.add_argument('--max_num_questions', default=8, type=int)
    parser.add_argument('--do_lower_case', action='store_true', default=False)
    parser.add_argument('--batch_size', '-b', default=64, type=int)
    parser.add_argument('--seed', default=42)
    parser.add_argument('--n_gpu', default=4)

    args = parser.parse_args()

    if args.output == 'none':
        args.ckpt = args.ckpt[:-1] if args.ckpt.endswith('/') else args.ckpt
        base_name = os.path.basename(args.ckpt)
        args.output = f'{os.environ["PJ_HOME"]}/results/fact_checking/AAAI22/{args.input}.{args.model_name_or_path}_m{args.mask_rate}_l{args.logic_lambda}_{base_name}_{args.prior}.predictions.jsonl'

    assert args.output.endswith('predictions.jsonl'), \
        f"{args.output} must end with predictions.jsonl"

    args.input = f'{os.environ["PJ_HOME"]}/data/fact_checking/v5/{args.input}.json'

    checker = FactChecker(args, args.ckpt, args.mask_rate)
    fever_results = checker.check_from_file(args.input, args.output, args.batch_size, args.verbose)
    cjj.lark(f"{args.output}: {fever_results}")
