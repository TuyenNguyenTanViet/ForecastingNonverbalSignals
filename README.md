# ForecastingNonverbalSignals

This is the code for the fact-sheet paper "Forecasting Nonverbal Socials during Dyadic Interactions".

## Usage

1. Install virtual environment named SocialActionGAN with dependencies: Python 3.6, TensorFlow 1.15, numpy, pickle5, sklearn, pandas, h5py

``` console
conda create -n SocialActionGAN python=3.6 tensorflow=1.15 pickle5 scikit-learn pandas h5py
```

2.  Download the preprocessed [UDVIA_2d.pickle] (https://drive.google.com/drive/folders/1I3xFvgljFxjImlNdR7qvP9M4AvgEI7Dc?usp=sharing) and put it inside /dataset. Training model with the default parameters:

``` console
(SocialActionGAN): python train.py
```

3. Download the pre-trained network (https://drive.google.com/drive/folders/1oohhV4Rryfw09Y1XMsBsS1bQorDx0HaC?usp=sharing) and put it inside /model. Forcasting the generated motions and generating the ouput file in the format of the challenge:

``` console
(SocialActionGAN): python generate.py --annotations_dir "/path_to/talk_annotations_test_masked/" --segments_path "/path_to/test_segments_topredict.csv"
```