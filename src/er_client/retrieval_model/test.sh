python test.py \
--test_path ../data/pages.json \
--bert_pretrain ../evidence_retrieval/bert_base \
--checkpoint ../evidence_retrieval/retrieval_model/model.best.pt \
--evi_num 5 \
--outdir ../data \
--name evidence.json
