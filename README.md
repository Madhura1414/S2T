# Our Contribution
Built a dataset that can used with this code or any other code in general.

Took the Initiative to build the dataset in kannada sign language with corresponding kannada text that will help in the training of the model.

## Dataset description

| Characteristic	| Data |
|-----------------|------|
|Word	| 36
|Sentence |	 16
|Character |	 61
|Videos 	|388
|Frame Rate |	30fps 
|Resolution |	1080 x 1920| 

## Feature Augmentation
Feature augmentation can help improve the generalization capability of a model. By augmenting the features, you introduce variations in the learned representations, which can enhance the model's ability to handle variations and improve its performance on unseen data. This can lead to better generalization and reduced overfitting.

In this work we prefered augmentations in the embedding space of deep neural networks, rather than as a pre-processing step in pixel-space. It identifies
minority class examples in the training set whose nearest neighbors contain adversary class members. SMOTE then takes certain point as base and creates k-means neighbours around it.  

Advantages of feature augmentation over pixel augmentation are:

1. No changes in training time. Since physical augmentation results in the increase of number of video samples, it will increase the training time.
2. Robustness to Noise: Feature augmentation techniques can be more robust to noise and distortions present in the data. By manipulating features, the  augmentation process is less affected by pixel-level noise.
   
CNN features embeddings are extracted from npy files. The arrays are given to the smote function in SMOTE_augmentation.py

## Result comaparison
|Dataset|	Feature Augmentation|	Pose Estimation Method	|Encoder	|Decoder	|Accuracy|
|-------|--------------------|-------------------------|---------|---------|--------|
|INCLUDE-50	 |         No|	OpenPose	|Mobile net V2|	BiLSTM	|73.9|
|Our Dataset	|         Yes|	Media pipe	| Mobile Net V2	 |BiLSTM	  |81.69|

## Train and Validation Accuracy
![image](https://github.com/Madhura1414/S2T/assets/84361102/df4e1ff6-4463-458b-83db-bb531fed217f)


## INLCUDE - Isolated Indian Sign Language Recognition

This repository contains code for training models on [INCLUDE](https://zenodo.org/record/4010759) dataset

# Dependencies

Install the dependencies through the following command

```bash
>> pip install -r requirements.txt
```



## Steps
- Download the INCLUDE dataset
- Run `generate_keypoints.py` to save keypoints from Mediapipe Hands and Blazepose for train, validation and test videos. 
```bash
>> python generate_keypoints.py --include_dir <path to downloaded dataset> --save_dir <path to save dir> --dataset <include/include50>
```
- Run `runner.py` to train a machine learning model on the dataset
```bash
>> python runner.py --dataset <include/include50> --use_augs --model transformer --data_dir <location to saved keypoints>
```
- Use the `--use_pretrained` flag to either perform only inference using pretrained model or resume training with the pretrained model. 
```bash
>> python runner.py --dataset <include/include50> --use_augs --model transformer --data_dir <location to saved keypoints> --use_pretrained <evaluate/resume_training>
```
- To get predictions for videos from a pretrained model, run the following command.
```bash
>> python evaluate.py --data_dir <dir with videos>
```

## Citation

```
@inproceedings{10.1145/3394171.3413528,
author = {Sridhar, Advaith and Ganesan, Rohith Gandhi and Kumar, Pratyush and Khapra, Mitesh},
title = {INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition},
year = {2020},
isbn = {9781450379885},
publisher = {Association for Computing Machinery},
doi = {10.1145/3394171.3413528},
numpages = {10},
series = {MM '20}
}
```
```
Dablain, Damien, et al. "Efficient augmentation for imbalanced deep learning." arXiv preprint arXiv:2207.06080 (2022).
```
