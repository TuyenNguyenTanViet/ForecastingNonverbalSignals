# ForecastingNonverbalSignals

This is the code for the fact-sheet paper "Forecasting Nonverbal Socials during Dyadic Interactions".

## Usage

1. Install virtual environment named  SocialActionGAN with dependencies: Python 3.6, TensorFlow 1.15, numpy, pickle5, sklearn, pandas, h5py
     ``` console
     conda create -n SocialActionGAN python=3.6 tensorflow=1.15 pickle5 scikit-learn pandas h5py
   ```
 2. Training model with the default parameters:
	  ``` console
	  (SocialActionGAN): python train.py
	   ```
 3. Forcasting the generated motions and generating the ouput file in the format of the challenge:
	  ``` console
	  (SocialActionGAN): python generate.py
	   ```


