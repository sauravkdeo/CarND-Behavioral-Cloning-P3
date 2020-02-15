import cv2
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import keras
import csv
from urllib.request import urlretrieve
import os
from sklearn.preprocessing import LabelBinarizer
from zipfile import ZipFile
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers import Lambda, Cropping2D


def uncompress_features_labels(dir,name):
    if(os.path.isdir(name)):
        print('Data extracted')
    else:
        with ZipFile(dir) as zipf:
            zipf.extractall('data')
            
def import_data(path):          
    samples = [] #simple array to append all the entries present in the .csv file

    with open(path) as csvfile: #currently after extracting the file is present in this path
        reader = csv.reader(csvfile)
        next(reader, None) #this is necessary to skip the first record as it contains the headings
        for line in reader:
            samples.append(line)
    return samples

def generator(samples, batch_size=32):
    num_samples = len(samples)
   
    while 1: 
        shuffle(samples) #shuffling the total images
        for offset in range(0, num_samples, batch_size):
            
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                    for i in range(0,3): #we are taking 3 images, first one is center, second is left and third is right
                        
                        name = './data/IMG/'+batch_sample[i].split('/')[-1]
                        center_image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB) #since CV2 reads an image in BGR we need to convert it to RGB since in drive.py it is RGB
                        center_angle = float(batch_sample[3]) #getting the steering angle measurement
                        images.append(center_image)
                        
                        # introducing correction for left and right images
                        # if image is in left we increase the steering angle by 0.2
                        # if image is in right we decrease the steering angle by 0.2
                        
                        if(i==0):
                            angles.append(center_angle)
                        elif(i==1):
                            angles.append(center_angle+0.2)
                        elif(i==2):
                            angles.append(center_angle-0.2)
                        
                        # Code for Augmentation of data.
                        # We take the image and just flip it and negate the measurement
                        
                        images.append(cv2.flip(center_image,1))
                        if(i==0):
                            angles.append(center_angle*-1)
                        elif(i==1):
                            angles.append((center_angle+0.2)*-1)
                        elif(i==2):
                            angles.append((center_angle-0.2)*-1)
                        #here we got 6 images from one image    
                        
        
            X_train = np.array(images)
            y_train = np.array(angles)
            
            yield sklearn.utils.shuffle(X_train, y_train) #here we do not hold the values of X_train and y_train instead we yield the values which means we hold until the generator is running
def create_model():
    model = Sequential()

    # Preprocess incoming data, centered around zero with small standard deviation 
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

    # trim image to only see section with road
    model.add(Cropping2D(cropping=((70,25),(0,0))))           

    #layer 1- Convolution, no of filters- 24, filter size= 5x5, stride= 2x2
    model.add(Convolution2D(24,5,5,subsample=(2,2)))
    model.add(Activation('elu'))

    #layer 2- Convolution, no of filters- 36, filter size= 5x5, stride= 2x2
    model.add(Convolution2D(36,5,5,subsample=(2,2)))
    model.add(Activation('elu'))

    #layer 3- Convolution, no of filters- 48, filter size= 5x5, stride= 2x2
    model.add(Convolution2D(48,5,5,subsample=(2,2)))
    model.add(Activation('elu'))

    #layer 4- Convolution, no of filters- 64, filter size= 3x3, stride= 1x1
    model.add(Convolution2D(64,3,3))
    model.add(Activation('elu'))

    #layer 5- Convolution, no of filters- 64, filter size= 3x3, stride= 1x1
    model.add(Convolution2D(64,3,3))
    model.add(Activation('elu'))

    #flatten image from 2D to side by side
    model.add(Flatten())

    #layer 6- fully connected layer 1
    model.add(Dense(100))
    model.add(Activation('elu'))

    #Adding a dropout layer to avoid overfitting. Here we are have given the dropout rate as 25% after first fully connected layer
    model.add(Dropout(0.25))

    #layer 7- fully connected layer 1
    model.add(Dense(50))
    model.add(Activation('elu'))


    #layer 8- fully connected layer 1
    model.add(Dense(10))
    model.add(Activation('elu'))

    #layer 9- fully connected layer 1
    model.add(Dense(1)) #here the final layer will contain one value as this is a regression problem and not classification


    # the output is the steering angle
    # using mean squared error loss function is the right choice for this regression problem
    # adam optimizer is used here
    model.compile(loss='mse',optimizer='adam')
    return model

uncompress_features_labels('/opt/carnd_p3/data.zip','data')
samples = import_data('./data/driving_log.csv')
print(len(samples))

train_samples, validation_samples = train_test_split(samples,test_size=0.1)
train_samples, test_samples = train_test_split(samples,test_size=0.1)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

my_model = create_model()

if os.path.isfile('model.h5'):
    my_model = keras.models.load_model('model.h5')

my_model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=5, verbose=1)

my_model.save('model.h5')

# my_model.evaluate(test_samples, verbose=0)
my_model.summary()
