# LegalBART

# Masking data for pre-training-

python legalBART/pre_training/prepare_pretrain_data.py

# Pre-training model-

python legalBART/pre_training/pretrain.py

# Generate augmentations-

python legalBART/augmentation_ner_qa/ner-aug.py

# Evaluate perplexity-

python legalBART/perplexity.py

Pre-training and fine-tuning uses the same technique, fine-tuning can be done by replacing the unsupervised dataset with supervised data.