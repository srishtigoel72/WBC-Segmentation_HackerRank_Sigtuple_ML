# WBC-Segmentation_HackerRank_Sigtuple_ML

This assingment will segment the WBC cells from the RBC cells.

Input : Image with RBC and WBC cells.  
Output : Mask highlightening the WBC region.

Requirements:

python 3.5

anaconda 4.2.0

tensorflow - conda

keras

theano


Structure of the Model:

A Convolutional Neural Network(CNN) is implemented having 5 convolution2d block of filter values 64, 32 and 1. 
Each layer is seperated by an activation layer(Relu).


Steps to follow:

1. Extract the data from wcbdata.zip and place the Train_Data and Test_Data folders in ./data/ folder.
2. Run Train.py to train and dump the model.
3. Run test.py to test the model on the images present in Test_Data folder and output masks are dumped in ./data/output folder
