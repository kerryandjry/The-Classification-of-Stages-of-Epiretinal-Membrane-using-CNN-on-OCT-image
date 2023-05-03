# The Classification of Stages of Epiretinal Membrane using Convolutional Neural Network on Optical Coherence Tomography Image

## Install dependencies
- pytorch
- numpy
- opencv
- matlibplot
- PIL
- tqdm

## Training data

## Run
### train
python train.py
### test
python predict.py

## Result

|     model   | train | validation |  test | params | 
| ----------- | ----- | ---------- | ----- | ------ |
| Resnet34 | 0.991 |    0.888   | 0.822 |    21M    |
| Mobilenet | 0.987 |   0.813  | 0.786 |    1.5M    |
| Efficientnet | 0.978 |  0.883  | 0.815 |   20M    | 
|  Swin Transformer | 0.988 | 0.947 | 0.811 |  28M  |
|   MIXER   | 0.991 |   0.931   | 0.818 |   7.4M    |

## Reference
https://github.com/ramprs/grad-cam //grad-cam
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html //ROC-Curve
