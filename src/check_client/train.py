# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import glob
import argparse
import logging
import random
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoConfig,
    AutoTokenizer
)
from transformers import WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup
import tensorflow as tf
from pytorch_lightning.loggers import WandbLogger

try:
    from .modules.data_processor import DataProcessor
    from .plm_checkers import BertChecker, RobertaChecker
    from .utils import init_logger, compute_metrics, set_seed
except:
    from modules.data_processor import DataProcessor
    from plm_checkers import BertChecker, RobertaChecker
    from utils import init_logger, compute_metrics, set_seed

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

mAutoModel = {
    'bert': BertChecker,
    'roberta': RobertaChecker,
}

logger = logging.getLogger(__name__)


def train(args, data_processor, model, tokenizer):
    """ Train the model """
    global wdblogger
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    tf.io.gfile.makedirs(os.path.dirname(args.output_dir))
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_dataset = data_processor.load_and_cache_data("train", tokenizer, args.data_tag)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  drop_last=True,
                                  batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("Num examples = %d", len(train_dataset))
    logger.info("Num Epochs = %d", args.num_train_epochs)
    logger.info("Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    tr_loss2, logging_loss2 = 0.0, 0.0
    tr_loss3, logging_loss3 = 0.0, 0.0
    set_seed(args)  # Added here for reproductibility
    model.zero_grad()
    for _ in range(int(args.num_train_epochs)):
        all_loss = 0.0
        all_accuracy = 0.0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "claim_input_ids": batch[0],
                "claim_attention_mask": batch[1],
                "qa_input_ids_list": batch[3],
                "qa_attention_mask_list": batch[4],
                "nli_labels": batch[-2],
                "labels": batch[-1],
            }
            if args.model_type != "distilbert":
                # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                inputs["claim_token_type_ids"] = batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                inputs["qa_token_type_ids_list"] = batch[5] if args.model_type in ["bert", "xlnet", "albert"] else None

            outputs = model(**inputs)
            loss, _loss2, logits = outputs[0], outputs[1], outputs[2]
            loss2, loss3 = _loss2

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
                loss2 = loss2.mean()
                loss3 = loss3.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                loss2 = loss2 / args.gradient_accumulation_steps
                loss3 = loss3 / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            tr_loss2 += loss2.item()
            tr_loss3 += loss3.item()

            all_loss += loss.detach().cpu().numpy() * args.gradient_accumulation_steps
            all_accuracy += np.mean(
                inputs["labels"].detach().cpu().numpy() == logits.detach().cpu().numpy().argmax(axis=-1)
            )
            description = "Global step: {:>6}, Loss: {:>.6f}, Accuracy: {:>.6f}".format(
                global_step,
                all_loss / (step + 1),
                all_accuracy / (step + 1),
            )
            epoch_iterator.set_description(description)
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log metrics
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well
                    if args.local_rank == -1 and args.evaluate_during_training:
                        results = evaluate(args, data_processor, model, tokenizer)
                        for key, value in results.items():
                            logger.warning(f"Step: {global_step}, eval_{key}: {value}")
                            wdblogger.log_metrics({"eval_{}".format(key): value}, global_step)
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    wdblogger.log_metrics({"lr": scheduler.get_lr()[0]}, global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    wdblogger.log_metrics({"loss": (tr_loss - logging_loss) / args.logging_steps}, global_step)
                    wdblogger.log_metrics({"loss2": (tr_loss2 - logging_loss2) / args.logging_steps}, global_step)
                    wdblogger.log_metrics({"loss3": (tr_loss3 - logging_loss3) / args.logging_steps}, global_step)

                    logging_loss = tr_loss
                    logging_loss2 = tr_loss2
                    logging_loss3 = tr_loss3
                    wdblogger.save()

                # Save model checkpoint
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break
        if 0 < args.max_steps < global_step:
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, data_processor, model, tokenizer, prefix=""):
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    dataset = data_processor.load_and_cache_data("eval", tokenizer, args.data_tag)
    eval_sampler = SequentialSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler,
                                 drop_last=True,
                                 batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("Num examples = %d", len(dataset))
    logger.info("Batch size = %d", args.eval_batch_size)

    label_truth, y_predicted, z_predicted, m_attn, mask = \
        do_evaluate(tqdm(eval_dataloader, desc="Evaluating"), model, args, during_training=True, with_label=True)

    outputs, results = compute_metrics(label_truth, y_predicted, z_predicted, mask)

    return results


def do_evaluate(dataloader, model, args, during_training=False, with_label=True):
    label_truth = []
    y_predicted = []
    z_predicted = []
    m_attn = []
    mask = []
    for i, batch in enumerate(dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {
                "claim_input_ids": batch[0],
                "claim_attention_mask": batch[1],
                "qa_input_ids_list": batch[3],
                "qa_attention_mask_list": batch[4],
                "nli_labels": batch[6],
            }
            
            if args.model_type != "distilbert":
                # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                inputs["claim_token_type_ids"] = batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                inputs["qa_token_type_ids_list"] = batch[5] if args.model_type in ["bert", "xlnet", "albert"] else None

            outputs = model(**inputs)

            if during_training and (i < 3 and (args.logic_lambda != 0)):
                logger.warning(f'* m_attn:\n {outputs[-2][:5]}\n')
                logger.warning(f'* Logic outputs:\n {outputs[-1][0][:5]}.\n Labels: {batch[-1][:5]}\n')

            if with_label:
                label_truth += batch[-1].tolist()
            y_predicted += outputs[2].tolist()
            mask += outputs[-1][1].tolist()
            z_predicted += outputs[-1][0].tolist()
            m_attn += outputs[-2].tolist()

    y_predicted = np.argmax(y_predicted, axis=-1).tolist()

    return label_truth, y_predicted, z_predicted, m_attn, mask


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(mAutoModel.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name",
    )
    parser.add_argument(
        "--data_tag",
        default='default',
        type=str,
        help='Tag to cached data'
    )
    parser.add_argument(
        "--max_seq1_length",
        default=None,
        type=int,
        required=True,
        help="The maximum total input claim sequence length after tokenization. "
             "Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_seq2_length",
        default=None,
        type=int,
        required=True,
        help="The maximum total input claim sequence length after tokenization. "
             "Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_num_questions",
        default=None,
        type=int,
        required=True,
        help='The maximum number of evidences.',
    )
    parser.add_argument(
        "--cand_k",
        default=1,
        type=int,
        help='The number of evidential answers out of beam size'
    )
    parser.add_argument(
        '--mask_rate',
        default=0.,
        type=float,
        help="Mask rate of QA"
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument('--logic_lambda', required=True, type=float,
                        help='Regularization term for logic loss, also an indicator for using only logic.')
    parser.add_argument('--prior', default='nli', type=str, choices=['nli', 'uniform', 'logic', 'random'],
                        help='type of prior distribution')
    parser.add_argument('--temperature', required=True, type=float, help='Temperature for gumbel softmax.')

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument(
        "--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    if args.do_train:
        global wdblogger
        tf.io.gfile.makedirs(args.output_dir)
        wdblogger = WandbLogger(name=os.path.basename(args.output_dir))
        wdblogger.log_hyperparams(args)
        wdblogger.save()
        log_file = os.path.join(args.output_dir, 'train.log')
        init_logger(logging.INFO if args.local_rank in [-1, 0] else logging.WARN, log_file)

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Prepare task
    data_processor = DataProcessor(
        args.model_name_or_path,
        args.max_seq1_length,
        args.max_seq2_length,
        args.max_num_questions,
        args.cand_k,
        data_dir=args.data_dir,
        cache_dir_name=os.path.basename(args.output_dir),
        overwrite_cache=args.overwrite_cache,
        mask_rate=args.mask_rate
    )

    # Make sure only the first process in distributed training will download model & vocab
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    # Load pretrained model and tokenizer
    args.model_type = args.model_type.lower()

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=3,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = mAutoModel[args.model_type].from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
        logic_lambda=args.logic_lambda,
        m=args.max_num_questions,
        prior=args.prior,
        temperature=args.temperature
    )

    # Make sure only the first process in distributed training will download model & vocab
    if args.local_rank == 0:
        torch.distributed.barrier()

    if args.do_train:
        model.to(args.device)
        wdblogger.watch(model)

    logger.info("Training/evaluation parameters %s", args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum
    # if args.fp16 is set. Otherwise it'll default to "promote" mode, and we'll get fp32 operations.
    # Note that running `--fp16_opt_level="O2"` will remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex
            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    # Training
    if args.do_train:
        global_step, tr_loss = train(args, data_processor, model, tokenizer)
        logger.info("global_step = %s, average loss = %s", global_step, tr_loss)

    # Save the trained model and the tokenizer
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs

        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = mAutoModel[args.model_type].from_pretrained(
                checkpoint,
                logic_lambda=args.logic_lambda,
                m=args.max_num_questions,
                prior=args.prior,
                temperature=args.temperature
            )
            model.to(args.device)

            # Evaluate
            result = evaluate(args, data_processor, model, tokenizer, prefix=global_step)
            result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
            results.update(result)

    print(results)
    return results


if __name__ == "__main__":
    main()
