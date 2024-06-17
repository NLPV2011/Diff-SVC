# Diff-SVC Refactor (Inference, training and model code simplifier and updated from RCell's [Diff-SVC](https://github.com/innnky/diff-svc/))
[diffsvc](https://github.com/prophesier/diff-svc) implemented based on [DiffSinger unofficial repository](https://github.com/keonlee9420/DiffSinger)

> It is still under development and testing, training and inference code are fully completed
> The conclusion of the temporary test is that when the number of people in the data set is too large (for example, 60 or 70 people), the sound leakage will be aggravated, and the sound leakage of about 5 people is basically the same as that of a single person\
> At present, you can see that there are a lot of branches, all of which are various solutions under testing \

## Introduction
Realize singing voice timbre conversion based on Diffsinger + softvc. Compared with the original diffsvc repository, this repository has the following advantages and disadvantages
+ Supports multiple speakers
+ This repository is based on the unofficial diffsinger repository, and the code structure is simpler and easier to understand
+ The vocoder also uses [441khz diffsinger community vocoder](https://openvpi.github.io/vocoders/)
+ Acceleration is not supported

Pre-downloaded files
+ softvc hubert (hubert-soft-0d54a1f4.pt) is placed in the hubert directory
+ 441khz diffsinger community vocoder (model) is placed in the hifigan directory
## Dataset preparation
You only need to put the dataset into the dataset_raw directory with the following file structure
```shell
dataset_raw
├───speaker0
│ ├───xxx1-xxx1.wav
│ ├───...
│ └───Lxx-0xx8.wav
└───speaker1
├───xx2-0xxx2.wav
├───...
└───xxx7-xxx007.wav
```

## Data preprocessing
Basically similar to sovits3.0
1. Resampling
```shell
python resample.py
```
2. Automatically divide training set, validation set and test set
```shell
python preprocess_flist_config.py
```
3. Generate hubert, f0, mel and stats
```shell
python preprocess_hubert_f0.py && python gen_stats.py
```

After executing the above steps, the dataset directory is the preprocessed data. You can delete the dataset_raw folder,
or delete the temporary wav file after resampling`rm dataset/*/*.wav`

## Training
```shell
python3 train.py --model naive --dataset ms --restore_step RESTORE_STEP
```

## Inference
[inference.py](inference.py)
