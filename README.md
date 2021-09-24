# ForecastingNonverbalSignals

  

This is the code for the fact-sheet "Forecasting Nonverbal Socials during Dyadic Interactions".

  

## Usage

  

1. Install virtual environment named SocialActionGAN with dependencies: Python 3.6, TensorFlow 1.15, numpy, pickle5, sklearn, pandas, h5py

``` console
conda create -n SocialActionGAN python=3.6 tensorflow=1.15 pickle5 scikit-learn pandas h5py
```

2.  Download [UDIVA_2d.pickle](https://drive.google.com/drive/folders/1I3xFvgljFxjImlNdR7qvP9M4AvgEI7Dc?usp=sharing), and put it in the folder [dataset](https://github.com/TuyenNguyenTanViet/ForecastingNonverbalSignals/tree/main/dataset). Training model with the default parameters:

``` console
(SocialActionGAN): python train.py
```

3. Download the [pre-trained model](https://drive.google.com/drive/folders/1oohhV4Rryfw09Y1XMsBsS1bQorDx0HaC?usp=sharing) and put it in the folder [model](https://github.com/TuyenNguyenTanViet/ForecastingNonverbalSignals/tree/main/model). Forecast the motions, generate the ouput file based on the format of the challenge:

``` console
(SocialActionGAN): python generate.py --annotations_dir "/path_to/talk_annotations_test_masked/" --segments_path "/path_to/test_segments_topredict.csv"
```

  

## Optional

1. Extract the training data, package it as UDIVA_2d.pickle:

``` console
(SocialActionGAN): python preprocessing.py --annotations_dir "/path_to/talk_annotations_train"
```