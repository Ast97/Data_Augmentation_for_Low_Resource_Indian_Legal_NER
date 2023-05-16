cd lstm-lm

python train.py \
  --train_file ../../NER_TRAIN/train_WK.lin.txt \
  --valid_file ../../NER_DEV/dev_WK.lin.txt \
  --model_file ../models/model_300_512.pt \
  --emb_dim 300 \
  --rnn_size 512