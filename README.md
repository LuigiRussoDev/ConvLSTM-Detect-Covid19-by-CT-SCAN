# ConvLSTM-Detect-Covid19-by-CT-SCAN
In this Repository I want show a new approach for detection COVID19 issue by CT SCAN images on sequences using a deep 
ConvLSTM approach. 

## Introduction
Recentely so many challeges have allowed AI enthusiasts and researchers to develop model of neural network for detection
of COVID19 disease by X-RAY or CT-SCAN images.
The main problem for this task (and in general for ML) consists in finding the datasets to allow the neural network model to train.
This neural network model was created for personal purposes and not to replace the figure of a doctor, but it could be considered as a 
good tool to support the doctor and in particular to allow science to be collaborative with other research institutes.
In this work I want to show a Model of Neural Network that consists to classify a CT-SCAN image covid or not. The model proposed is 
based on <b> ConvLSTM </b> on a particular dataset of CT-SCAN that describe a sequences of image in different timestamps. 


## Model Description
The Neural Network model used for solve this task is based on ConvLSTM in 2D. The main reason of this type of recurrent neural network is 
based to capture spatio-temporal correlation with convolutional structures in several phases of the process: input-to-state and 
state-to-state transitions.
<p float="left">
  <img src="img/sequences/proc_img_28.tif" width="100" />
  <img src="img/sequences/proc_img_29.tif" width="100" /> 
  <img src="img/sequences/proc_img_30.tif" width="100" />
</p>
