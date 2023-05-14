# MULDA

# Linearize Data -

python MULDA/code/ner_scripts/linearize_ner.py

# SPM Encode the Data -

python MULDA/code/ner_scripts/spm_encode.py

# FairSeq Preprocess the Data -

python MULDA/code/mbart_fairseq_copy/fairseq/fairseq_cli/preprocess.py  --source-lang eng  --target-lang eng  --trainpref './MULDA/data/indian_ner_legal_dataset/spm_encoded/train_spm' --validpref './MULDA/data/indian_ner_legal_dataset/spm_encoded/val_spm' --testpref './MULDA/data/indian_ner_legal_dataset/spm_encoded/test_spm' --destdir './MULDA/data/indian_ner_legal_dataset/fairseq_preprocess/' --thresholdtgt 0 --thresholdsrc 0 --srcdict './MULDA/code/mbart_fairseq_copy/fairseq/mbart.cc25.v2/dict.txt' --tgtdict './MULDA/code/mbart_fairseq_copy/fairseq/mbart.cc25.v2/dict.txt' --workers 10

# Finetune mBART
python MULDA/code/mbart_fairseq_copy/fairseq/fairseq_cli/train.py './MULDA/data/indian_ner_legal_dataset/fairseq_preprocess/' --arch mbart_large --layernorm-embedding --task translation_from_pretrained_bart --source-lang eng --target-lang eng --criterion label_smoothed_cross_entropy --label-smoothing 0.2 --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' --lr-scheduler polynomial_decay --lr 5e-05 --warmup-updates 1000 --total-num-update 550000 --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 --max-tokens 24000 --update-freq 2  --save-interval 2 --no-epoch-checkpoints --seed 222 --log-format simple --log-interval 2  --restore-file './MULDA/code/mbart_fairseq_copy/fairseq/mbart.cc25.v2/' --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler --langs eng --ddp-backend no_c10d  --max-epoch 2 --batch-size 2

# Generate Samples
python MULDA/code/mbart_fairseq_copy/fairseq/fairseq_cli/generate.py './MULDA/data/indian_ner_legal_dataset/fairseq_preprocess/' --path './MULDA/code/mbart_fairseq_copy/fairseq/mbart.cc25.v2/model.pt' --task translation_from_pretrained_bart --gen-subset test -s eng -t eng --bpe 'sentencepiece' --sentencepiece-model './MULDA/code/mbart_fairseq_copy/fairseq/mbart.cc25.v2/sentence.bpe.model' --sacrebleu --remove-bpe 'sentencepiece' --batch-size 2 --langs ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN --unkpen 2 --beam 5 > './MULDA/data/indian_ner_legal_dataset/generate_data/generate.txt'


