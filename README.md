## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)

[image1]: ./germanTrafficSign/50SpeedLimit.jpg "SpeedLimit of 50"
[image2]: ./germanTrafficSign/noentry.jpg "No entry"
[image3]: ./germanTrafficSign/stop.jpg "Stop"
[image4]: ./germanTrafficSign/straightOnly.jpg "Ahead Only"
[image5]: ./germanTrafficSign/workOnRoad.jpg "Road Work"
[image6]: ./germanTrafficSign/yield.jpg "Yield"

Project Overview
---
<!--- ![alt text][image1] --->

In this project, you will use what you've learned about deep neural networks and convolutional neural networks to classify traffic signs. You will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, you will then try out your model on images of German traffic signs that you find on the web.

We have included an Ipython notebook that contains further instructions 
and starter code. Be sure to download the [Ipython notebook](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb). 

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

## The Project
---
The goals / steps of this project are the following:
### Load the data set
---
I have used pickle library to read the data from the already preprocessed data given to us.


### Explore, summarize and visualize the data set
---
Followed by which I have simply looked at the data size of the training and testing data set along with the image size to resructure the deep network accordingly. Each image proved out to be 32x32x3, thus I first resized the images to a gray scale image to attemp to use Lenet architecture initially and to then slowly build from there.

We can see a lot of variation in the frequency of training samples of respective classes with frequency of different classes given as :  
``` 
[ 180 1980 2010 1260 1770 1650  360 1290 1260 1320 1800 1170 1890 1920
  690  540  360  990 1080  180  300  270  330  450  240 1350  540  210
  480  240  390  690  210  599  360 1080  330  180 1860  270  300  210
  210 ]
```

where class 0 'Speed limit (20km/h)' with 180 samples, class 1 'Speed limit (30km/h)' with 1980 samples and so on and so forth.

### Design, train and test a model architecture
As I mentioned above the first step of data preprocessing was to change the image to gray scale using 
```
X = 0.2989*X[:,:,:,0] + 0.5870*X[:,:,:,1] + 0.1140*X[:,:,:,2]
```
Ref : https://www.mathworks.com/matlabcentral/answers/196535-function-to-convert-rgb-to-grayscale

Then I trained the network on the Lenet to se the training and validation accurary, which was around 89%, as mentioned in the lecture notes. To gain more accuracy the first thing that I thought of is that in comparison to LeNet now we have much more classes to clasify 43 in comparison to 10, which pointed to the requirement of more parameters. To make sure I am not overtraining over my data set since I was adding bunch of extra parameters I added dropout layers after each layer.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray Scale image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, output 28x28x18 	|
| RELU					|												|
| Dropout	      	| Keep Probabiligy = 0.80 				|
| Max pooling	      	| 2x2 stride,  output 14x14x18 				|
|     |       |
| Convolution 3x3	    | 1x1 stride, valid padding, output 10x10x18 	|
| RELU					|												|
| Dropout	      	| Keep Probabiligy = 0.80 				|
| Max pooling	      	| 2x2 stride,  output 5x5x18 				|
|     |       |
| Flatten             | output 450x1  |
| Fully connected		| output 240x1 							|
| RELU					|												|
| Dropout	      	| Keep Probabiligy = 0.80 				|
|     |       |
| Fully connected		| output 129x1 							|
| RELU					|												|
| Dropout	      	| Keep Probabiligy = 0.80 				|
|     |       |
| Fully connected		| output 43x1 							|
| RELU					|												|
| Dropout	      	| Keep Probabiligy = 0.80 				|


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

