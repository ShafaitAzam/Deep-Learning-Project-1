# Height Estimation of Passenger using Spectrograms of Sonar Signal In TensorFlow CNN
This is a project for Machine Learning course of Information Technology under Frankfurt University of Applied Sciences supervised by Dr. Prof. Andreas Pech
# Contributors:
If you have any query related to this project please contact two of the Authors:

Shafait Azam(email- shafait.azam@stud.fra-uas.de) - Data Preprocessing, Data Modelling, Algorithm Deployment, Core Programmer.

Mashnunul Huq(email- mashnunul.huq@stud.fra-uas.de) - Data Acquisition, Programmer, Automation.
# Key Words
Raw Data retrieval from FIUS Red Pitaya Ultrasonic Sensor System, Time Series Data Analysis, Data Preprocessing, Data Modelling, Convolutional Neural Network Deployment, Passenger's Height Estimation
# Data Repository
As the data file is big the ADC data retrieved from two SONAR sensors attached with Radpitaya can be downloaded from below links
Lower sensor: https://drive.google.com/file/d/1zcM6RURsS3SvCEv5lUTWFKrbN8u_Uw2H/view?ts=643eb1ca
Upper sensor: https://drive.google.com/file/d/1Jxn6NIRaSjZDK_I2HiUXpHcbwpzgeb2a/view?ts=643eb1f2
# How to Train Machines
There are two machines for two sensors to detect if there is a person present in the seat or not. So it is problem of binary classification. The main logic behind this project was to train two machines for two of the sensors (upper and lower) to detect if there is a full sized human present covering both upper and lower sensors or there is a small infant covering only the lower sensor or there is no one in the seat (explained in details in the documentation section).
First download two csv files for data acquired from the ADC signal of SONAR supported Radpitaya system. Then use the portion of data preprocessing to generate spectrograms. Here data file location have to be given as argument while calling "DataPreProcessing" class and by calling "save_Spectogram" method of that class we can save the spectrograms and numpy arrays of those spectograms into two seperated folders. Here spectograms will help Human Eye to understand scenarios of the whole spectograms and numpy arrays of these spectograms are easier for computer to read for the next process. We have taken only the first 512 values of the frequency spectogram to take only significant changes into account of a whole time series signal for reducing computational hazards.


After doing the data preprocessing only by calling the class CNNMachine one can easily make the CNN machines with predefined parameters for 50k data for a project without changing any internal parameters. While initiating this class one must give the true csv file and the spectogram folder's path. after training by only calling the "predict" method one can easily get the result of prediction from a new data set if there is a person present before the sensor or not.
# Accuracy
As we have splitted the data set into 85% traning data set and 15% testing dataset, we got a lowest accuracy of 99.8% after several runs in different processors and kaggle.  
# Usage of The Project 
The purpose of this project is to get the weight for a CNN machine to prepare for an array of SONAR sensors where each detection of each sensor can tell a certain height deference of the passenger or driver of the seat. For this reason detection of presence of a person is required which has been done in this project with satisfactory accuracy level. With the detection of the height that car seat can be automatically adjusted for proper view for the driver and the passenger.     
