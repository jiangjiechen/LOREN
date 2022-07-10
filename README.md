# LOREN

Resources for our AAAI 2022 paper: "[LOREN: Logic-Regularized Reasoning for Interpretable Fact Verification](https://ojs.aaai.org/index.php/AAAI/article/view/21291)".

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

If you find our paper or resources useful to your research, please kindly cite our paper.

```latex
@article{Chen_Bao_Sun_Zhang_Chen_Zhou_Xiao_Li_2022, 
  title={LOREN: Logic-Regularized Reasoning for Interpretable Fact Verification}, 
  volume={36}, 
  url={https://ojs.aaai.org/index.php/AAAI/article/view/21291}, DOI={10.1609/aaai.v36i10.21291}, 
  abstractNote={Given a natural language statement, how to verify its veracity against a large-scale textual knowledge source like Wikipedia? Most existing neural models make predictions without giving clues about which part of a false claim goes wrong. In this paper, we propose LOREN, an approach for interpretable fact verification. We decompose the verification of the whole claim at phrase-level, where the veracity of the phrases serves as explanations and can be aggregated into the final verdict according to logical rules. The key insight of LOREN is to represent claim phrase veracity as three-valued latent variables, which are regularized by aggregation logical rules. The final claim verification is based on all latent variables. Thus, LOREN enjoys the additional benefit of interpretability --- it is easy to explain how it reaches certain results with claim phrase veracity. Experiments on a public fact verification benchmark show that LOREN is competitive against previous approaches while enjoying the merit of faithful and accurate interpretability. The resources of LOREN are available at: https://github.com/jiangjiechen/LOREN.}, 
  number={10}, 
  journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
  author={Chen, Jiangjie and Bao, Qiaoben and Sun, Changzhi and Zhang, Xinbo and Chen, Jiaze and Zhou, Hao and Xiao, Yanghua and Li, Lei}, 
  year={2022}, 
  month={Jun.}, 
  pages={10482-10491} 
}
```
