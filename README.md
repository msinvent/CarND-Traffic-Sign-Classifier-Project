## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Project Overview
---
In this project, you will use what you've learned about deep neural networks and convolutional neural networks to classify traffic signs. You will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, you will then try out your model on images of German traffic signs that you find on the web.

We have included an Ipython notebook that contains further instructions 
and starter code. Be sure to download the [Ipython notebook](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb). 

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

### Minimal Project Requirement : 
---
* the Ipython notebook with the code
* the code exported as an html file
* a writeup report either as a markdown or pdf file 

## The Project
---
The goals / steps of this project are the following:
### Load the data set
---
I have used pickle library to read the data from the already preprocessed data given to us. And, initialized three different variables X_train, X_Valid, X_test for training, validation and testing purposes from the three corresponding data files.

### Explore, summarize and visualize the data set
---
Followed by which I have simply looked at the data size of the training and testing data set along with the image size to resructure the deep network accordingly. Each image proved out to be 32x32x3, thus I first resized the images to a gray scale image to attemp to use Lenet architecture initially and to then slowly build from there.

### Design, train and test a model architecture
As I mentioned above the first step of data preprocessing was to change the image to gray scale using 
X = 0.2989*X[:,:,:,0] + 0.5870*X[:,:,:,1] + 0.1140*X[:,:,:,2]
Ref : https://www.mathworks.com/matlabcentral/answers/196535-function-to-convert-rgb-to-grayscale

Then I trained the network on the Lenet to se the training and validation accurary, which was around 89%, as mentioned in the lecture notes. To gain more accuracy the first thing that I thought of is that in comparison to LeNet now we have much more classes to clasify 43 in comparison to 10, which pointed to the requirement of more parameters. To make sure I am not overtraining over my data set since I was adding bunch of extra parameters I added dropout layers after each layer.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray Scale image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, output 28x28x18 	|
| RELU					|												|
| Dropout	      	| Keep Probabiligy = 0.80 				|
| Max pooling	      	| 2x2 stride,  output 14x14x18 				|
| Convolution 3x3	    | 1x1 stride, valid padding, output 10x10x18 	|
| RELU					|												|
| Dropout	      	| Keep Probabiligy = 0.80 				|
| Max pooling	      	| 2x2 stride,  output 5x5x18 				|
| Flatten             | output 450x1  |
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

### Dataset and Repository

1. Download the data set. The classroom has a link to the data set in the "Project Instructions" content. This is a pickled dataset in which we've already resized the images to 32x32. It contains a training, validation and test set.
2. Clone the project, which contains the Ipython notebook and the writeup template.
```sh
git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
cd CarND-Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```

