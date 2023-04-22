#Imports
import cv2
import glob
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import scipy
from datetime import datetime
from typing import Tuple

import tensorflow as tf
from tensorflow import keras

from keras import backend as K
from keras.layers import Input, Lambda, Dense, Flatten, Dropout
from keras.models import Model, Sequential
from keras.applications import vgg16, ResNet50, resnet, densenet, DenseNet169, DenseNet201, ResNet101, ResNet152
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import image_dataset_from_directory
from keras import optimizers
from keras.losses import Huber


from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.utils import shuffle
from sklearn.metrics import r2_score

#Ignore future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None # default = 'warn'

#GLOBAL PARAMS
PRED_COLUMN = 'width'

class Regression_Creator():
    '''
    Input: d_befone_spline.csv
    Output: saves altered images to drive with the regression values to their filenames
    '''
    def __init__(self) -> None:
        pass

class DF_Creator:

    def __init__(self, dir_path='C:\\Users\\victo\\Programming\\l-ded_ml_models\\images\\images_xpos', save_path=None) -> None:
        #Sets up the directories for the data to be extracted from
        self.DATADIR = []
        for directory in os.listdir(dir_path):
            self.DATADIR.append(dir_path + '\\' + str(directory))

        #Create main dataframe and loop over paths in the DATADIR to append 
        self.df = pd.DataFrame()
        for path in self.DATADIR:
            self.df = self.df.append(self.preprocess_pipeline(path))

        #Reset the index and extract the width and x pos columns from the filename
        self.df.reset_index(inplace=True, drop=True)
        self.extract_from_filename() #modifies the dataframe inplace

        if save_path is not None:
            self.save_df(save_path)

    #Return the dataframe for each of the standoff distances folder
    def preprocess_pipeline(self, path) -> pd.DataFrame:
        #Grab paths for all .tiff images and write
        glob_path = path + '/*.tiff'

        #Create dataframe with the img_path column from globbing the .tiff files in the path
        df = pd.DataFrame(columns=['img_path'])
        df['img_path'] = glob.glob(glob_path)

        #Create the standoff distance labels from the folder the image was from
        labels = [float(path.split('\\')[-1])] * df.shape[0]
        df['standoff distance'] = labels

        return df

    #Alters the dataframe inplace
    def extract_from_filename(self, xpos=True, width=True) -> None:
        self.df['x pos'] = [np.NaN] * self.df.shape[0]
        self.df['width'] = [np.NaN] * self.df.shape[0]

        #Iterate over a copy of the dataframe and extract the values from the images filenames
        iter_df = self.df.copy()
        for index, row in iter_df.iterrows():

            #Get path from the row
            path_str = row['img_path']
            path_split = path_str.split('_')
            
            #Find the width from the filename
            width = path_split[-1].split('.')[0]
            width_float = float(width)/(10**3)

            #Same for x position
            str_xpos = str(path_split[-2])
            x_pos_float = float(str_xpos.replace('-', '.').strip())
            
            #preprocess standoff distance
            standoff_dist = float(row['standoff distance']) / 100
            
            #Make changes to the df
            self.df.loc[index, 'standoff distance'] = standoff_dist
            self.df.loc[index, 'width'] = width_float
            self.df.loc[index, 'x pos'] = x_pos_float

    #Saves the dataframe to the save_path
    def save_df(self, save_path):
        self.df.to_csv(save_path)

class DF_Reader:

    def __init__(self, filepath='D:\\Users\\Victor\\Área de Trabalho\\train_history\\df\\df_xpos_images.csv', shuffle=True, reduce=None) -> None:
        self.df = pd.read_csv(filepath, index_col=0)

        #Reset index and drop null values
        self.df.reset_index(drop=True, inplace=True)
        self.df.dropna(subset='width', inplace=True)
        
        #If use samples dataset
        if reduce is not None:
            self.df = self.reduce_dataset(reduce)
        
        #Create train test dfs
        self.train_df, self.test_df = self.create_train_test_dfs()

        #Shuffle train test dfs inplace
        self.shuffle_train_test_dfs()

    #Create the train, test dataframes
    def create_train_test_dfs(self) -> Tuple:
        train_df = pd.DataFrame()
        test_df = pd.DataFrame()

        #Create test columns and make sure they are balanced
        for index, inner_df in self.df.groupby(by='standoff distance'):
            inner_train_df, inner_test_df = train_test_split(inner_df, test_size=0.2)
            train_df = train_df.append(inner_train_df)
            test_df = test_df.append(inner_test_df)

        #Reset indexes of train and test datasets
        train_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)

        return train_df, test_df

    #Shuffle the dataframes
    def shuffle_train_test_dfs(self):
        self.train_df = shuffle(self.train_df)
        self.test_df = shuffle(self.test_df)
        self.df = shuffle(self.df)

    def reduce_dataset(self, reduce) -> pd.DataFrame:
        sampled_df = self.df.sample(n=reduce)
        sampled_df.reset_inedx(drop=True, inplace=True)
        return sampled_df

#Read the dataframe from memory if it was already created once
def read_df_drom_memory(filepath) -> pd.DataFrame:
    #df = pd.read_csv('D:/Users/Victor/Área de Trabalho/train_history/df' + '/df_new_xpos_images.csv', index_col=0)
    df = pd.read_csv(filepath, index_col=0)
    return df

class CNN_model():
    def __init__(self) -> None:
        pass

    def create_datagens(self):
        pass

    def create_feature_extraction_generator(self):
        pass

    def create_generators(self):
        pass

    def create_model(self):
        pass