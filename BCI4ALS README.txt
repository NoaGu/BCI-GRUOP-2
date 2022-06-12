# BCI4ALS- Team 2- Headset XX
## README


This is the code repository for the BCI4ALS, team 2, with headset XX.
The code is a fork of Asaf Harel(harelasa@post.bgu.ac.il) basic code for the course BCI-4-ALS which
was taken place in the Hebrew University during 2021/2022. You are free to use, change, adapt and
so on - but please cite properly if published. We assume you have already set up a Matlab
environment with libLSL, OpenBCI, EEGLab with ERPLAB & loadXDF plugins installed. 


- Noa Guttman (noa.guttman@mail.huji.ac.il)
- Uri Stern (uri.stern@mail.huji.ac.il)
- Nadav Kaduri (nadav.kaduri@mail.huji.ac.il)


## Project Structure


The repository is structured into 3 directories:


- Offline- Matlab code used for offline training.
- Online- Matlab code used for online training.
- Recordings- Recording from the headset on our mentor.


### Offline


This part of the code is responsible for recording raw EEG data from the headset, preprocess it, segment it, extract features and
train a classifier.


1. MI1_Training.m- Code for recording new training sessions.
2. MI2_Preprocess.m- Function to preprocess raw EEG data.
3. MI3_SegmentData.m- Function that segments the preprocessed data into chunks.
4. MI4_ExtractFeatures.m- Function to extract features from the segmented chunks.
5. kfoldtest.m- Function to train a classifier based on the features extracted earlier.
6. Additional matlab files for helpful subfunctions of the code. Specifically, we’ll mention connectruns.mat, which connects data from different recordings.




### Online


This part of the code is responsible for loading the classifier that was trained in the offline section, record raw EEG data, preprocess it, extract features and
make a prediction using the classifier. Additionally, it saves the features to use for training later (for possible co-adaptive learning in the future) and presents the prediction to the user on the screen.


1. MI_Online_Learning.m- A script used to run and give feedback online. 
2. PreprocessBlock.m- Simillar to the offline phase, this function preprocess online chunk.
3. MI4_featureExtraction_online2.m- Simillar to the offline phase, this function extract features from the preprocessed chunk.
4. prepareTraining.m- Prepare a training vector for co-learning.


### data
In the recordings folder, we have 4 types of recorded data from our mentor:
1. Imagery 1-4: recordings of motor imagery with 2 classes (moving left and right hands)
2. Imagery 4 classes: single recording of motor imagery with 4 classes (moving left and right hands and legs)
3. Motor ex 1-2: recordings of motor execution with 2 classes (moving left and right hands)
4. Online 1-2: online recordings of error signal (mentor reaction for the prediction presented in the online phase)
Each recording consists of X trials, on 13 channels (channel 11 is problematic, but we didn’t remove it).




So how do we use this code?
Offline:
First, we open Matlab, OpenBCI and labrecorder. Now we start:
1. Open MI1_Training.m, read the documentation and change parameters as needed. Most importantly, change
   where to save the training vector(rootFolder). The training vector is a vector containing the labels for each trial.
2. Next, open OpenBCI and start a session, don't forget to configure the network widget correctly.
3. Run MI1_Training.m and follow the console instructions.
4. Change the output dir of the lab recorder(File name/Template) to the directory created automatically in step 3.
   (If you can't change the output dir, make sure BIDS is not checked). Name the file EEG.XDF.
5. Update the lab recorder streams, select both of them(eeg and marker), and start recording. Make sure the status bar in the bottom shows an increasing
   number of data(KBs recieved).
6. Continue to training.
7. To process the data run NI2,MI3,MI4 and then kfoldtest. 
Online:
First, we open Matlab, OpenBCI and labrecorder.
1. open MI_Online_lerneing.m, start running
2. Change the output dir of the lab recorder(File name/Template)
3. Continue run MI_Online_lerneing.m and follow the console instructions.