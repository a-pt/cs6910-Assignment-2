# CS6910-Assignment-2 [CS20M005, CS20M016]

Part A we have implemented a code that implements Convolutional Neural Network on a subset of iNaturalist dataset with the following functionalities.
-
```
Optimizers               :Stochastic,Momentum,Adam
Activation Fucntions     :ReLu, Tanh
Initialization           :Random
Number of Filters        :16, 32
Stride values            :1, 2
```
We have also used the following as the hyperparameters.
-

*Dropout*                 :The Dropout layer randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting
*Batch Normalization*     :Batch normalization significantly reduces training time by normalizing the input of each layer in the network. In practical coding, we add Batch   
                           Normalization after the activation function of the output layer or before the activation function of the input layer. 
*Filter Organization*     :It will organize the number of filters in each layer of CNN.
*Data Augmentation*       :It is a technique to artificially create new training data from existing training data.

At dense layer we have used "softmax" as the activation function.

The implementation is linked with wandb and hyper parameter tuning can be done effectively by changing the values of sweep confiiguration in the script. The configuration used for parameter searching are as follows.
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

