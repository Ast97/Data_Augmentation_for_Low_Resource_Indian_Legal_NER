cd lstm-lm

python generate.py \
--model_file ../models/model_300_512.pt \
--out_file ../samples/samples_50000_300_512.txt \
--num_sentences 50000 \
--temperature 1.0 \
--seed 3435 

echo "Generated samples"

cd ../tools

python line2cols.py \
--inp_file ../samples/samples_50000_300_512.txt \
--out_file ../samples/out_50000_300_512.txt

echo "Cleaned samples by removing inconsistent generated tags"