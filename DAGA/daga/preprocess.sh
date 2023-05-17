cd tools

python joinjsons.py \
--in_file1 ../../NER_TRAIN/NER_TRAIN_PREAMBLE.json \
--in_file2 ../../NER_TRAIN/NER_TRAIN_JUDGEMENT.json \
--out_file ../../NER_TRAIN/train.json

python joinjsons.py \
--in_file1 ../../NER_DEV/NER_DEV_PREAMBLE.json \
--in_file2 ../../NER_DEV/NER_DEV_JUDGEMENT.json \
--out_file ../../NER_DEV/dev.json

echo "Joined Json files for train and dev"

python preprocess_wikiann.py \
--in_file ../../NER_TRAIN/train.json \
--out_file ../../NER_TRAIN/train_WK.txt

python preprocess_wikiann.py \
--in_file ../../NER_DEV/dev.json \
--out_file ../../NER_DEV/dev_WK.txt

echo "Converted data to wikiann format"

python preprocess.py  \
 --train_file ../../NER_TRAIN/train_WK.txt \
 --dev_file ../../NER_DEV/dev_WK.txt

echo "Linearised dataset"