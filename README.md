# LOREN

Resources for our AAAI 2022 paper (pre-print): "[LOREN: Logic-Regularized Reasoning for Interpretable Fact Verification](https://arxiv.org/abs/2012.13577)".

![front](https://github.com/jiangjiechen/LOREN/blob/main/docs/front.png)


## DEMO System

Check out our [demo system](https://huggingface.co/spaces/Jiangjie/loren-fact-checking)! 
Note that the results will be slightly different from the paper, since we use an up-to-date Wikipedia as the evidence source whereas [FEVER](https://fever.ai) uses Wikipedia dated 2017.

## Dependencies

- CUDA > 11
- Prepare requirements: `pip3 install -r requirements.txt`.
  - Also works for `allennlp==2.3.0, transformers==4.5.1, torch==1.8.1`.
- *Set environment variable* `$PJ_HOME`: `export PJ_HOME=/YOUR_PATH/LOREN/`.

## Download Pre-processed Data and Checkpoints

- **Pre-processed data** at [Google Drive](https://drive.google.com/file/d/1kZxHidaDCe5GWMIuNa5fVeK4LPYCpvDD/view?usp=sharing). Unzip it and put them under `LOREN/data/`.
  - Data for training a Seq2Seq MRC is at `data/mrc_seq2seq_v5/`.
  - Data for training veracity prediction is at `data/fact_checking/v5/*.json`.
    - *Note*: `dev.json` uses *ground truth evidence* for validation, where `eval.json` uses *predicted evidence* for validation. This is consistent with the settings in KGAT.
  - Evidence retrieval models are not required for training LOREN, since we directly adopt the retrieved evidence from [KGAT](https://github.com/thunlp/KernelGAT), which is at `data/fever/baked_data/` (using only during pre-processing).
  - Original data is at `data/fever/` (using only during pre-processing). 

- **Pre-trained checkpoints** at [Huggingface Models](https://huggingface.co/Jiangjie/loren). Unzip it and put them under `LOREN/models/`.
  - Checkpoints for veracity prediciton are at `models/fact_checking/`.
  - Checkpoint for generative MRC is at `models/mrc_seq2seq/`.
  - Checkpoints for KGAT evidence retrieval models are at `models/evidence_retrieval/` (not used in training, displayed only for the sake of completeness).

## Training LOREN from Scratch

*For quick training and inference **with pre-processed data & pre-trained models**, please go to [Veracity Prediction](#2-Veracity-Prediction).*

First, go to `LOREN/src/`.

### 1 Building Local Premises from Scratch

#### 1) Extract claim phrases and generate questions

*You'll need to download three external models in this step, i.e., two models from AllenNLP in `parsing_client/sentence_parser.py` and a T5-based question generation model in `qg_client/question_generator.py`. Don't worry, they'll be automatically downloaded.*

- Run `python3 pproc_client/pproc_questions.py --roles eval train val test` 
- This generates cached json files:
  - `AG_PREFIX/answer.{role}.cache`: extracted phrases are stored in the field `answers`.
  - `QG_PREFIX/question.{role}.cache`: generated questions are stored in the field `cloze_qs`, `generate_qs` and `questions` (two types of questions concatenated).

#### 2) Train Seq2Seq MRC

##### Prepare self-supervised MRC data (only for SUPPORTED claims)

- Run `python3 pproc_client/pproc_mrc.py -o LOREN/data/mrc_seq2seq_v5`.
- This generates files for Seq2Seq training in a [HuggingFace](https://github.com/huggingface/transformers) style:
  - `data/mrc_seq2seq_v5/{role}.source`: concatenated question and evidence text.
  - `data/mrc_seq2seq_v5/{role}.target`: answer (claim phrase).

##### Training Seq2Seq

- Go to `mrc_client/seq2seq/`, which is modified based on HuggingFace's examples.
- Follow `script/train.sh`.
- The best checkpoint will be saved in `$output_dir` (e.g., `models/mrc_seq2seq/`).
  - Best checkpoints are decided by ROUGE score on dev set.

#### 3) Run MRC for all questions and assemble local premises

- Run `python3 pproc_client/pproc_evidential.py --roles val train eval test -m PATH_TO_MRC_MODEL/`.
- This generates files:
  - `{role}.json`: files for veracity prediction. Assembled local premises are stored in the field `evidential_assembled`.

#### 4) Building NLI prior

Before training veracity prediction, we'll need a NLI prior from pre-trained NLI models, such as [DeBERTa](https://huggingface.co/docs/transformers/model_doc/deberta).

- Run `python3 pproc_client/pproc_nli_labels.py -i PATH_TO/{role}.json -m microsoft/deberta-large-mnli`.
- ***Mind the order!*** The predicted classes [Contradict, Neutral, Entailment] correspond to [REF, NEI, SUP], respectively.
- This generates files:
  - Adding a new field `nli_labels` to `{role}.json`.

### 2 Veracity Prediction

This part is rather easy (less pipelined :P). A good place to start if you want to skip the above pre-processing.

#### 1) Training

- Go to folder `check_client/`.
- See what `scripts/train_*.sh` does.

#### 2) Testing

- Stay in folder `check_client/`
- Run `python3 fact_checker.py --params PARAMS_IN_THE_CODE`
- This generates files:
  - `results/*.predictions.jsonl`

#### 3) Evaluation

- Go to folder `eval_client/`
- For **Label Accuracy** and **FEVER score**: `fever_scorer.py`

- For **CulpA** (turn on `--verbose` in testing): `culpa.py`

## Citation

If you find our paper or resources useful to your research, please kindly cite our paper (pre-print, official published paper coming soon).

```latex
@misc{chen2021loren,
      title={LOREN: Logic-Regularized Reasoning for Interpretable Fact Verification}, 
      author={Jiangjie Chen and Qiaoben Bao and Changzhi Sun and Xinbo Zhang and Jiaze Chen and Hao Zhou and Yanghua Xiao and Lei Li},
      year={2021},
      eprint={2012.13577},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
