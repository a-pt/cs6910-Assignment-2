# CS6910-Assignment-2 [CS20M005, CS20M016]

Part A
-
We have implemented a code that implements Convolutional Neural Network on a subset of iNaturalist dataset with the following functionalities..<br/><br/>
```
Optimizers               :Stochastic,Momentum,Adam
Activation Fucntions     :ReLu,Tanh
Initialization           :Random
Number of Filters        :16,32
Stride values            :1,2
```
We have also used the following as the hyperparameters.<br/><br/>

* *Dropout*                 : The Dropout layer randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting.<br/><br/>
* *Batch Normalization*     : Batch normalization significantly reduces training time by normalizing the input of each layer in the network. In practical coding, we add Batch 
                             Normalization after the activation function of the output layer or before the activation function of the input layer.<br/><br/>
* *Filter Organization*     : It will organize the number of filters in each layer of CNN.<br/><br/>
* *Data Augmentation*       : It is a technique to artificially create new training data from existing training data.<br/><br/>

At dense layer we have used "softmax" as the activation function.<br/><br/>
The implementation is linked with wandb and hyper parameter tuning can be done effectively by changing the values of sweep confiiguration in the script. The configuration used for parameter searching are as follows.<br/><br/>
```
'epoch': [5,7,8,10]
'dropout': [0.1,0.2,0.3]
 'number_of_filters': [16,32]
'filter_organisation': [0.5,1,2]
'size_filter': [3,4]
'data_augmentation': ['Yes','No']
'batch normalization': ['Yes','No']
'learning_rate': [0.001,0.0001]
'optimizer_fn': ['adam','sgd','momentum']
'activation_fn': ['relu','tanh']
'dense': [128,256,512]
'stride': [1,2]
```
Q1&Q2.iypnb implements a CNN from scratch and find best hyperparameters using wanbd sweeps.
Q4&Q5.iypng Visualises the best model detected using guided backpropogation, Grad-CAM and guided Grad-CAM


Part B
-
We have implemented a code that performs Fine Tuning on a pre-trained model. We have loaded the some models which are pre-trained on ImageNet dataset, we have used ImageNet dataset as it is somehow similar to iNaturalist Dataset<br/><br/>
We have also used the following functionalities while training.<br/><br/>
```
Pre-Trained Model        :Inception,Resnet,Inceptionresnet,Xception
Optimizers               :Adam,Stochastic
Activation Fucntions     :ReLu,Tanh
```
We have also used the following as the hyperparameters.<br/><br/>
* K_ft : the layer number from which convolution starts unfreezing for fine tuning.<br/><br/>
* Ft_bool : If set Yes, that means we have to do fine tuning.<br/><br/>
* In_epochs : No of epochs to run with pre trained model.<br/><br/>
* Ft_epoch : No. of epochs to run after unfreezing of layers to fine tune them.<br/><br/>
* preprocess_input : to adequate the input to the format the model requires.<br/><br/>
* Global Average Pooling : An operation that calculates the average output of each feature map in the previous layer. This fairly simple operation reduces the data significantly and prepares the model for the final classification layer. It also has no trainable parameters, just like Max Pooling.<br/><br/>

Here too at dense layer we have used "softmax" as the activation function.<br/><br/>
Similar to  Part A, here the implementation is linked with wandb and hyper parameter tuning can be done effectively by changing the values of sweep confiiguration in the script. The configuration used for parameter searching are as follows.<br/><br/>
```
'base': ['inception','resnet','inceptionresnet','xception']
'in_epoch': [4,5,10]
'ft_epoch': [4,5,10]
'ft_bool': ['Yes']
'dropout': [0.1,0.2]
'optimizer_fn': ['adam','sgd']
'activation_fn': ['relu','tanh']
'dense': [256,512]
```
Q1-3.iypnb implements wandb sweeps to findout best hyper parametes for pre trained model and do fine tuning based on those.

Part C
-
We have used a pre-trained model named as YoloV3 in this part for object detection in an image as well as in a video.<br/><br/>
od_image.iypnb implements object detection from images.
Object_Detection_in_Video_.ipynb implements object detection from videos.
hd_video.py implements real time human detection using YoloV3 and OpenCV
