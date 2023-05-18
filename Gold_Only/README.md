# Gold Only

The NER model (legalBERT) is trained using the gold data from the Indian Legal NER dataset without any data augmentation. This is the main baseline model on which the data augmented samples generated from our legalBART, MulDa and DAGA models will be evaluated on. Around 70\% of the total gold data (train + validation + test) was assigned to the training dataset to train this model. Around 20\% of the total data was assigned to the validation dataset. The remaining 10\% was reserved for testing (unseen data). We used Ray Tune to find the optimal model hyperparameters for training the model. 

Gold Data: 
Train data - https://github.com/Ast97/Data_Augmentation_for_Low_Resource_Indian_Legal_NER/tree/main/Data_preprocess/NER_TRAIN
Use NER_TRAIN_JUDGEMENT_PREPROCESSED.json and NER_TRAIN_PREAMBLE_PREPROCESSED.json
Test data - https://github.com/Ast97/Data_Augmentation_for_Low_Resource_Indian_Legal_NER/tree/main/Data_preprocess/NER_DEV
Use NER_DEV_JUDGEMENT_PREPROCESSED.json and NER_DEV_PREAMBLE_PREPROCESSED.json

Model Training and Evaluation steps:
1. Hyperparameter tuning: https://github.com/Ast97/Data_Augmentation_for_Low_Resource_Indian_Legal_NER/blob/main/Gold_Only/Gold_Only_Indian_Legal_NER_Hyperparameter_Tuning.ipynb
2. Training and Evaluation: https://github.com/Ast97/Data_Augmentation_for_Low_Resource_Indian_Legal_NER/blob/main/Gold_Only/Training_Gold_Only_and_Evaluating_Data_Augmented_Samples.ipynb 
 # Note:
 I used reference from https://medium.com/@andrewmarmon/fine-tuned-named-entity-recognition-with-hugging-face-bert-d51d4cb3d7b5 blog to tokenize my datasets (tokenize_and_align_labels()).
