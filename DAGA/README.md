
# DAGA: Data Augmentation with Generative Approach

This repository is cloned from https://github.com/ntunlp/daga and all the code from 'lstm-lm' directory is used as is. 

## Environment Setup

To set up the project environment, follow the steps below:

1. Create a new conda environment with Python 3.7:
   ```
   conda create --name daga python=3.7
   ```

2. Activate the conda environment:
   ```
   conda activate daga
   ```

3. Install the required packages using pip:
   ```
   pip install torch==1.2.0 torchvision==0.4.0 torchtext==0.5
   ```

## Preprocess

Before training the model, you need to preprocess the data. The preprocessing step prepares the data for training. To preprocess the data, execute the following command:

```bash
sh preprocess.sh
```

## Training

Once the data is preprocessed, you can start training the DAGA model. To initiate the training process, run the following command:

```bash
sh train.sh
```

## Generate

After the training is complete, you can generate new samples using the trained DAGA model. To generate new samples, execute the following command:

```bash
sh generate.sh
```

The output samples will be stored in the following location: `./DAGA/daga/samples/daga_samples_postprocessed.csv`