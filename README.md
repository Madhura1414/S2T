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

