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

#Keras, tensorflow imports
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

#General sklearn imports
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.utils import shuffle
from sklearn.metrics import r2_score

#Spline imports
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
import matplotlib.image

#Ignore future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None # default = 'warn'

#GLOBAL PARAMS
PRED_COLUMN = 'width'
X_COL = 'img_path'

class Regression_Creator:
    '''
    Input: d_befone_spline.csv
    Output: saves altered images to drive with the regression values to their filenames
    '''

    def __init__(self) -> None:
        self.bef_reg = pd.read_csv('C:\\Users\\victo\\Programming\\l-ded_ml_models\\files\\d_before_spline.csv', index_col=0)

    def plot_spline(self, x_plot, y_plot, x_true, y_true, label, save=False, show=False):
        #Plot curves
        plt.plot(x_plot, y_plot, color='orange')
        plt.scatter(x_true, y_true, color='black')
        
        #Save file to save location
        if save:            
            #Try creating dir to save spline images to
            self.splines_save_dir = f'C:\\Users\\victo\\Programming\\l-ded_ml_models\\spline_plots\\plots_{self.degree}degrees_{self.knots}knots'
            try:
                os.mkdir(self.splines_save_dir)
            except:
                pass

            #Setup filename
            filename = self.splines_save_dir + f'\\spline_label_{int(np.round(label))}_{np.random.randint(low=0, high=1000000)}.jpg'

            #Save images
            plt.savefig(filename)

        #Show figure of splines or not
        if show:
            plt.show()
        else:
            plt.close()

    def invert_second_line(self):
        invert = self.bef_reg.copy(deep=True)

        prev_index = -1
        count = 0
        invert['line loc'] = [np.NaN] * invert.shape[0]
        for index, row in invert.iterrows():
            
            #First iteration
            if prev_index == -1:
                prev_index = index
                invert.loc[index, 'line loc'] = count
                continue

            if index != prev_index + 1:
                count += 1

            invert.loc[index, 'line loc'] = count
            prev_index = index

        new_df = pd.DataFrame()
        #Now, we have a dataframe that has the values for each 
        for label_name, df in invert.groupby(by='label'):
            for line_loc, line_df in df.groupby(by='line loc'):
                if line_loc % 2 == 0:
                    new_df = pd.concat([new_df, line_df])
                    continue
                
                img_path_list = line_df.iloc[::-1, 0].tolist()
                img_path_list_reversed = reversed(img_path_list)
                line_df.loc[:, 'img_path'] = img_path_list_reversed

                #Invert all columns and append to new dataframe
                new_df = pd.concat([new_df, line_df])

        #Save new df to csv 
        new_df.to_csv('invert.csv')
    
    def apply_spline_regression(self, degree, knots, alpha=1e-3, save=False, show=False):
        self.degree = degree
        self.knots = knots

        #Create a copy of the dataframe before regression
        self.after_reg = self.bef_reg.copy(deep=True)

        #Loop over the dataframe for each label
        for label_name, reg_df in self.bef_reg.groupby(by='label'):
            spline_df = reg_df[~reg_df['local'].isnull()][['x pos', 'value', 'id for reg']]
            
            #define training points for spline
            x_train = spline_df['x pos'].to_numpy()
            y_train = spline_df['value'].to_numpy()
            
            #Reshape for input on pipeline
            X_train = x_train[:, np.newaxis]
            Y_train = y_train[:, np.newaxis]
            
            #Define and fit spline model on training data
            spline_model =  make_pipeline(SplineTransformer(n_knots=knots, degree=degree), Ridge(alpha=alpha))
            spline_model.fit(X_train, Y_train)
            
            #Prediction
            x_pred = reg_df.loc[~reg_df['value'].isnull()]['x pos'].to_numpy()
            X_pred = x_pred[:, np.newaxis] #Reshape for prediction
            y_pred = spline_model.predict(X_pred) #Predict
            
            #plots
            x_plot =  np.linspace(0, 140, 280)
            X_plot = x_plot[:, np.newaxis]
            y_plot = spline_model.predict(X_plot)

            #Save plots
            self.plot_spline(X_plot, y_plot, x_train, y_train, label_name, save=save, show=show)
            
            #write to main dataframe based on the indexes and the spline predictions
            index_list = reg_df.loc[~reg_df['value'].isnull()].index.tolist()
            for index, reg_value in zip(index_list, y_pred.tolist()):
                self.after_reg.loc[index, 'reg value'] = reg_value

    def modify_reg_df_for_saving(self):
        #Create new columns for modifying file names later
        self.after_reg['split_img_path'] = self.after_reg['img_path'].str.split('\\')
        self.after_reg['filename_no_tiff'] = self.after_reg['split_img_path'].apply(lambda x: x[-1].split('.')[0])

        #Function to apply to each element in the reg value column
        def convert_str_width(x):
            if np.isnan(x):
                return str(x)
            
            x_times_million = float(x)*(10**6)
            x_int = int(np.round(x_times_million))
            return str(x_int).split('.')[0]
        
        #Apply function to convert regression value to str and multiply by 1 million, for reading purposes later
        self.after_reg['width str'] = self.after_reg['reg value'].apply(convert_str_width)
        self.after_reg.loc[73754, 'width str'] #seems good

        #Function to convert x_pos column to str, extra functionality so it always has at least 3 digits after the '_'
        def create_x_pos_str(x):
            x = str(x)
            x_split = x.split('.')
            x_0 = x_split[0]
            x_1 = x_split[1][:3]
            if x_1 == '0':
                x_1 = '000'
            
            return str(x_0) + '-' + str(x_1)
        self.after_reg['x pos str'] = self.after_reg['x pos'].apply(create_x_pos_str)

        #it works so lol, just creates a new column with a lot of string additions
        self.after_reg['filename'] = self.after_reg['filename_no_tiff'] + (['_'] * self.after_reg.shape[0]) + self.after_reg['x pos str'].astype('str') + (['_'] * self.after_reg.shape[0]) + self.after_reg['width str'].astype('str') + (['.tiff'] * self.after_reg.shape[0])

    def save_new_images(self):
        self.modify_reg_df_for_saving()
        #Create folders based on the label column, save images to that folder based on their respective labels
        main_path = os.path.join(os.getcwd(), f'images\\{self.degree}degrees_{self.knots}knots')

        try:
            os.mkdir(main_path)
        except:
            print("Directory already exists, images already saved for this degree and knots")
            return

        #Create folders based on all labels, astype int and then str to remove any decimal digits from the folder name
        for folder in self.after_reg['label'].astype('int32').astype('str').unique():
            folder_path = os.path.join(main_path, folder)
            try:
                os.mkdir(folder_path)
            except:
                continue

        for index, row in self.after_reg.iterrows():
            #create folder path, try accessing the folder name and write it if you read an error
            filename = str(row['filename'])
            label_folder = str(int(row['label']))
            save_path = main_path + '\\' + label_folder + '\\' + filename
            
            #with the folders created, save the images inside them (this takes around 10-15 mins)
            array = cv2.imread(row['img_path'] ,cv2.IMREAD_GRAYSCALE)
            array_cut = array[135:305 , 130:300]
            matplotlib.image.imsave(save_path, array_cut, cmap='gray')

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

    def __init__(self, filepath='C:\\Users\\victo\\Programming\\l-ded_ml_models\\files\\df_xpos_images.csv', shuffle=True, reduce=None) -> None:
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
        #self.df = shuffle(self.df)

    def reduce_dataset(self, reduce) -> pd.DataFrame:
        sampled_df = self.df.sample(n=reduce)
        sampled_df.reset_index(drop=True, inplace=True)
        return sampled_df

#This class creates the ImageDataGenerator generators
class Pipeline:

    def __init__(self, data_augmentation=False) -> None:
        self.data_augment = data_augmentation

    def set_cnn_pipeline(self, preprocessing_function, train_df, test_df, target_size, batch_size, feature_extraction) -> None:
        self.feature_extraction = feature_extraction
        self.preprocessing_function = preprocessing_function

        if self.feature_extraction:
            self.feature_extraction_datagen = self.create_datagens()
            self.train_feature_generator, self.test_feature_generator = self.create_feature_extraction_generators(train_df, test_df, target_size, batch_size)
        else:
            self.train_datagen, self.test_datagen = self.create_datagens()
            self.train_generator, self.valid_generator, self.test_generator = self.create_generators(train_df, test_df, target_size, batch_size)

    def create_datagens(self, rescale=(1/255.0), val_split=0.2) -> Tuple:
        #If extract features from the CNN
        if self.feature_extraction:
            feature_extraction_datagen = ImageDataGenerator(
                rescale = rescale,
                preprocessing_function = self.preprocessing_function)
            return feature_extraction_datagen

        #Train datagen
        #Augment data for the training dataset
        if self.data_augment:
            train_datagen = ImageDataGenerator(
                rescale = rescale,
                validation_split = val_split,
                preprocessing_function = self.preprocessing_function,
                rotation_range = 120,
                horizontal_flip = True,
                vertical_flip = True)
        else:
            train_datagen = ImageDataGenerator(
                rescale = rescale,
                validation_split = val_split,
                preprocessing_function = self.preprocessing_function)
        
        #Test datagen
        test_datagen = ImageDataGenerator(
            rescale = rescale,
            preprocessing_function = self.preprocessing_function)    
        return train_datagen, test_datagen

    def create_feature_extraction_generators(self, train_df, test_df, target_size, batch_size):
        train_feature_generator = self.feature_extraction_datagen.flow_from_dataframe(
            dataframe = train_df,
            x_col = X_COL,
            y_col = PRED_COLUMN,
            target_size = target_size,
            class_mode = 'raw',
            color_mode = 'rgb',
            shuffle = False
        )

        test_feature_generator = self.feature_extraction_datagen.flow_from_dataframe(
            dataframe = test_df,
            x_col = X_COL,
            y_col = PRED_COLUMN,
            target_size = target_size,
            class_mode = 'raw',
            color_mode = 'rgb',
            shuffle = False
        )
        return train_feature_generator, test_feature_generator

    def create_generators(self, train_df, test_df, target_size, batch_size) -> Tuple:
        train_generator = self.train_datagen.flow_from_dataframe(
            dataframe = train_df.drop(columns=['standoff distance', 'x pos']),
            x_col = X_COL,
            y_col = PRED_COLUMN,
            target_size = target_size,
            batch_size = batch_size,
            subset ='training',
            class_mode = 'raw',
            color_mode = 'rgb',
            shuffle=False
        )
        valid_generator = self.train_datagen.flow_from_dataframe(
            dataframe = train_df.drop(columns=['standoff distance', 'x pos']),
            x_col = X_COL,
            y_col = PRED_COLUMN,
            target_size = target_size,
            batch_size = batch_size,
            class_mode = 'raw',
            color_mode = 'rgb',
            subset = 'validation',
            shuffle=False
        )
        test_generator = self.test_datagen.flow_from_dataframe(
            dataframe = test_df.drop(columns=['standoff distance', 'x pos']),
            x_col = X_COL,
            y_col = PRED_COLUMN,
            target_size = target_size,
            batch_size = batch_size,
            class_mode = 'raw',
            color_mode = 'rgb',
            shuffle=False
        )
        return train_generator, valid_generator, test_generator

class CNN_model:
    def __init__(self, filepath='C:\\Users\\victo\\Programming\\l-ded_ml_models\\files\\df_xpos_images.csv') -> None:
        self.df_reader = DF_Reader(filepath=filepath)
        self.df = self.df_reader.df
        self.train_df = self.df_reader.train_df
        self.test_df = self.df_reader.test_df

    def create_model(self, base_model, custom_layers):
        #Make it so the base model layers are untrainable
        base_model.trainable = False

        #Create the new model
        self.model = Sequential()
        self.model.add(base_model)

        #Add custom layers to model
        for custom_layer in custom_layers:
            self.model.add(custom_layer)

        return self.model

    def create_layers(self, layers):
        if layers == None:
            return None
        
        l = []
        for layer in layers:
            layer_type = layer[0]
            if layer_type == 'flatten':
                cur_layer = Flatten()
            elif layer_type == 'dense':
                cur_layer = Dense(layer[1], activation=layer[2])
            elif layer_type == 'dropout':
                cur_layer = Dropout(rate=layer[1])
            else:
                raise Exception("no layer named flatten, dense or dropout")
            l.append(cur_layer)
            #del(cur_layer)
        return l

    def grid_train(self, params_grid, layers, pooling, cnn_pipeline):
        histories = []
        evaluations = []
        predictions = []

        print('params grid:', params_grid)

        for params in params_grid.values():
            #Print statistics
            print()
            print('input shape:', params['shape'])
            print('optimizer', params['optimizer'])
            print('loss function', params['loss function'])
            print('learning rate', params['learning rate'])
            print('batch size', params['batch size'])

            model, history, evaluation, prediction = self.train(params, layers=layers, pooling=pooling, pipeline=cnn_pipeline)

            #Append to list
            histories.append(history)
            evaluations.append(evaluation)
            predictions.append(prediction)

        return histories, evaluations, predictions

    def base_model_mapping(self, base_model_str, pooling, input_shape_tuple):
        if base_model_str == 'vgg16':
            base_model = vgg16.VGG16(weights='imagenet', include_top=False, pooling=pooling, input_shape=input_shape_tuple)
            preprocessing_function = vgg16.preprocess_input
        elif base_model_str == 'resnet50':
            base_model = ResNet50(weights='imagenet', include_top=False, pooling=pooling, input_shape=input_shape_tuple)
            preprocessing_function = resnet.preprocess_input
        elif base_model_str == 'resnet101':
            base_model = ResNet101(weights='imagenet', include_top=False, pooling=pooling, input_shape=input_shape_tuple)
            preprocessing_function = resnet.preprocess_input
        elif base_model_str == 'resnet152':
            base_model = ResNet152(weights='imagenet', include_top=False, pooling=pooling, input_shape=input_shape_tuple)
            preprocessing_function = resnet.preprocess_input
        elif base_model_str == 'densenet169':
            base_model = DenseNet169(weights='imagenet', include_top=False, pooling=pooling, input_shape=input_shape_tuple)
            preprocessing_function = densenet.preprocess_input
        elif base_model_str == 'densenet201':
            base_model = DenseNet201(weights='imagenet', include_top=False, pooling=pooling, input_shape=input_shape_tuple)
            preprocessing_function = densenet.preprocess_input
        
        else:
            raise Exception("base model not supported")

        return base_model, preprocessing_function

    def train(self, params, layers, pooling, pipeline):
        #Create the model
        base_model, preprocessing_function = self.base_model_mapping(params['base model'], pooling, params['shape'])
        model = self.create_model(base_model, self.create_layers(layers))

        #Create the generators based on the input shape
        pipeline.set_cnn_pipeline(preprocessing_function, self.train_df, self.test_df, params['shape'][:2], params['batch size'], feature_extraction=False)
        train_steps, valid_steps, test_steps = pipeline.train_generator.samples // params['batch size'], pipeline.valid_generator.samples // params['batch size'], pipeline.test_generator.samples // params['batch size']

        #Optimizer and loss function
        if params['optimizer'] == 'adam':
            optimizer = optimizers.Adam(learning_rate=params['learning rate'])
        elif params['optimizer'] == 'rms':
            optimizer = optimizers.RMSprop(learning_rate=params['learning rate'])

        #Loss function definition
        loss_function = params['loss function']
        if loss_function == 'huber_loss':
            loss_function = Huber(delta=5.0)

        #Adaptive learning rate
        my_lr_scheduler = keras.callbacks.LearningRateScheduler(adapt_learning_rate)
        
        #Compile the model
        model.compile(optimizer=optimizer, loss=loss_function, metrics=['mae', 'mse', tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsolutePercentageError(), r2])
        model.summary()

        #Train the model
        history = model.fit(pipeline.train_generator, 
                            epochs = params['epochs'], 
                            batch_size = params['batch size'],
                            steps_per_epoch = train_steps, 
                            validation_data = pipeline.valid_generator, 
                            validation_steps = valid_steps, 
                            callbacks=[my_lr_scheduler], verbose=1)
        
        #Evaluate the model
        evaluation = model.evaluate(pipeline.test_generator,
                                    batch_size = params['batch size'],
                                    verbose = 'auto',
                                    steps = test_steps,
                                    return_dict=True)
        
        prediction = model.predict(pipeline.test_generator,
                                batch_size = params['batch size'])

        #save model
        self.save_history(history, evaluation, prediction, params, model)
            
        return model, history, evaluation, prediction

    def save_history(self, history, evaluation, prediction, params, model):

        #Create directory for all information to be stored in
        datetime_str = str(datetime.now().strftime('%Y-%m-%d_%H_%M_%S'))
        directory = 'D:\\Users\\Victor\\Área de Trabalho\\train_history\\history_' + datetime_str
        os.mkdir(directory)

        #Save history as pkl file to directory
        self.save_pickle_history(history, directory)

        #Save dataframe with the model parameters
        self.save_model_params(params, directory)

        #Save model summary txt
        self.save_model_summary(model, directory)

        #Save mse, mae and r2 plots
        self.save_mse_plot(history, directory)
        self.save_mae_plot(history, directory)
        self.save_r2_plot(history, directory)

        #Create eval_results_df
        eval_tuples = list(zip(self.test_df['standoff distance'], self.test_df['width'], self.test_df['x pos'], prediction))
        self.eval_results_df = pd.DataFrame(eval_tuples, columns=['standoff distance', 'true', 'x pos', 'pred'])

        #Fix the prediction column which comes inside a list for some fucking reason
        self.eval_results_df['pred'] = self.eval_results_df['pred'].apply(lambda x: x[0])

        #save eval results df to disk
        self.eval_results_df.to_csv(directory + '\\eval_results_df.csv')

        #Create other plots
        self.save_true_vs_pred_plot(directory)
        self.save_mean_true_vs_pred_plot(directory)
        self.save_width_over_x_plot(directory, adjusted=True)

    def save_pickle_history(self, history, directory):
        pickle_filename = directory + '\\history.pkl'
        with open(pickle_filename , 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

    def save_model_params(self, params, directory):
        print(f"params: {params}")
        try:
            model_params_df = pd.DataFrame.from_dict(params, orient='columns')
            model_params_df.drop(columns='layers', inplace=True)
            model_params_df.to_csv(directory + '\\model_params.csv')
        except:
            print('Error on saving model params')

    def save_model_summary(self, model, directory):
        summary_path = directory + '\\modelsummary.txt'
        with open(summary_path, 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))

    def save_mse_plot(self, history, directory):
        mse_plot_name = '\\history_mse_plot.png'
        mse_plot_path = directory + mse_plot_name
        plt.plot(history.history['mse'], label='mse')
        plt.plot(history.history['val_mse'], label='val_mse')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Squared Error')
        plt.ylim([0, np.median(history.history['mse']) * 3])
        plt.legend(loc='lower right')
        plt.savefig(mse_plot_path)
        plt.close()

    def save_mae_plot(self, history, directory):
        mae_plot_name = '\\history_mae_plot.png'
        mae_plot_path = directory + mae_plot_name
        plt.plot(history.history['mae'], label='mae')
        plt.plot(history.history['val_mae'], label='val_mae')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Average Error')
        plt.ylim([0, np.median(history.history['mae']) * 3])
        plt.legend(loc='lower right')
        plt.savefig(mae_plot_path)
        plt.close()

    def save_r2_plot(self, history, directory):
        r2_plot_name = '\\history_r2_plot.png'
        r2_plot_path = directory + r2_plot_name
        plt.plot(history.history['r2'], label='r2_score')
        plt.plot(history.history['val_r2'], label='val_r2_score')
        plt.xlabel('Epoch')
        plt.ylabel('R2 Score')
        plt.ylim([0, 1])
        plt.legend(loc='lower right')
        plt.savefig(r2_plot_path)
        plt.close()

    def save_true_vs_pred_plot(self, directory):
        true_vs_pred_plot_name = '\\true_vs_pred_lineplot.png'
        true_vs_pred_plot_path = directory + true_vs_pred_plot_name

        #Create the true vs pred plot
        sns.scatterplot(data=self.eval_results_df, x='true', y='pred', s=5, color='b')
        sns.lineplot(x=np.arange(self.eval_results_df['true'].min(), self.eval_results_df['true'].max()), y=np.arange(self.eval_results_df['true'].min(), self.eval_results_df['true'].max()), color='black')
        
        #Define lower, upper bounds for x and y
        xy_min = min(self.eval_results_df['true'].min(), self.eval_results_df['pred'].min())
        xy_max = max(self.eval_results_df['true'].max(), self.eval_results_df['pred'].max())
        plt.xlim([xy_min, xy_max])
        plt.ylim([xy_min, xy_max])
        
        #Save and show figure
        plt.savefig(true_vs_pred_plot_path)
        plt.close()

    def save_mean_true_vs_pred_plot(self, directory):
        standoff_x_plot_name = '\\standoff_x_lineplot.png'
        standoff_x_plot_path = directory + standoff_x_plot_name

        standoff_grouped_true = self.eval_results_df.groupby('standoff distance')['true', 'pred'].mean()
        sns.scatterplot(data=self.eval_results_df, x='standoff distance', y='pred', s=5, color='b', label='predicted points')
        sns.lineplot(data=standoff_grouped_true, x=standoff_grouped_true.index, y='true', color='g', label='true mean')
        sns.lineplot(data=standoff_grouped_true, x=standoff_grouped_true.index, y='pred', color='black', label='pred mean')
        plt.legend(loc='lower right')
        plt.savefig(standoff_x_plot_path)
        plt.close()

    def save_width_over_x_plot(self, directory, adjusted=False):
        standoff_distances_folder = directory + '\\plots_standoff_distance'
        os.mkdir(standoff_distances_folder)

        for standoff_distance, df in self.eval_results_df.groupby(by='standoff distance'):
            df.sort_values(by='x pos', inplace=True, ascending=True)
            
            plt.figure(figsize=(20,6))
            plt.ylim([600,1050])
            
            sns.pointplot(data=df, x='x pos', y='true', color='black', label='true')
            sns.pointplot(data=df, x='x pos', y='pred', color='orange', label='pred')
            
            plt.xticks(rotation=45)
            plt.locator_params(axis='x', nbins=5)
            plt.title(f'standoff distance: {standoff_distance}')
            plt.legend(loc='lower right')

            #save each file to folder
            curr_standoff_dist_plot_path = f'\\{standoff_distance}.png'
            plt.savefig(standoff_distances_folder + curr_standoff_dist_plot_path)
            plt.close()
        
        if adjusted:
            for standoff_distance, df in self.eval_results_df.groupby(by='standoff distance'):
                df.sort_values(by='x pos', inplace=True, ascending=True)

                #figure configs
                plt.figure(figsize=(20,6))
                plt.ylim([-550,550])

                #calculate adjusted y values
                true_over = df['true'].to_numpy() / 2
                true_under = df['true'].to_numpy() / -2
                pred_over = df['pred'].to_numpy() / 2
                pred_under = df['pred'].to_numpy() / -2

                #plot data
                sns.lineplot(x=np.linspace(0, 135.33, df.shape[0]).astype('int'), y=true_over, color='black', label='true over')
                sns.lineplot(x=np.linspace(0, 135.33, df.shape[0]).astype('int'), y=pred_over, color='orange', label='pred over')
                sns.lineplot(x=np.linspace(0, 135.33, df.shape[0]).astype('int'), y=true_under, color='black', label='true under')
                sns.lineplot(x=np.linspace(0, 135.33, df.shape[0]).astype('int'), y=pred_under, color='orange', label='pred under')

                #plot configs
                plt.locator_params(axis='x', nbins=13)
                plt.legend()
                plt.title(f'standoff distance: {standoff_distance}')
                
                #save each file to folder
                curr_standoff_dist_plot_path = f'\\scaled_{standoff_distance}.png'
                plt.savefig(standoff_distances_folder + curr_standoff_dist_plot_path)
                plt.close()

class SVR_model:

    def __init__(self, features_df) -> None:
        self.features_df = shuffle(features_df)

    def create_train_test_dfs(self) -> None:
        self.train_df, self.test_df = train_test_split(self.features_df, shuffle=False, test_size=0.2)
        
        self.X_train = self.train_df.iloc[:, 4:]
        self.Y_train = self.train_df.loc[:, 'width']
        self.X_test = self.test_df.iloc[:, 4:]
        self.Y_test = self.test_df.loc[:, 'width']
    
    def scale_features(self):
        print(f"before scaling:", self.X_train.iloc[0,0])
        self.scaler = StandardScaler()
        self.X_train = pd.DataFrame(self.scaler.fit_transform(self.X_train))
        self.X_test = pd.DataFrame(self.scaler.fit_transform(self.X_test))
        print(f"after scaling:", self.X_train.iloc[0,0])

    def pca(self, plot=False):
        pca = PCA()
        pca.fit(self.X_train)

        accumulated_variance = np.array([np.sum(pca.explained_variance_ratio_[:i]) for i in range(self.X_train.shape[1])])

        if plot:
            plt.figure(figsize=(20,5))
            plt.plot(range(1, len(accumulated_variance)+1), accumulated_variance, 'o-', linewidth=2, color='blue')
            plt.title('Scree Plot')
            plt.xlabel('Principal Component')
            plt.ylabel('Variance Explained')
            plt.show()
        
        n_components = np.argmax(accumulated_variance > 0.975)
        reduction = (1 - (n_components / self.X_train.shape[1])) * 100
        print(f"initial number of features: {self.X_train.shape[1]}, n_pca_components: {n_components} ,feature reduction %: {reduction:.3f}")
        
        print('Before pca:')
        print('X_train shape:', self.X_train.shape)
        print('X_test shape:', self.X_test.shape)

        self.pca = PCA(n_components = n_components)
        self.X_train = self.pca.fit_transform(self.X_train)
        self.X_test = self.pca.transform(self.X_test)

        print('After pca:')
        print('X_train_pca shape:', self.X_train.shape)
        print('X_test_pca shape:', self.X_test.shape)

    def run_svr(self, kernel='rbf', C=1.0, gamma='scale', epsilon=0.1) -> None:
        self.svr = SVR(kernel=kernel, C=C, gamma=gamma, epsilon=epsilon, verbose=True)
        self.svr.fit(self.X_train, self.Y_train)

    def score_svr(self):
        self.r_squared = self.svr.score(self.X_test, self.Y_test)
        print(f"r2: {self.r_squared}, svr params: {self.svr.get_params()}")

    def predict_svr(self, save_csv=False):
        predictions = self.svr.predict(self.X_test)
        self.test_df['predictions'] = predictions

        #print(f"test_df columns", self.test_df.columns, 'head:', self.test_df.iloc[:10, :10])
        if save_csv:
            self.test_df.to_csv('predictions_csv.csv')

    def plot_results(self):
        pass

class Feature_extractor:

    def __init__(self, filepath='C:\\Users\\victo\\Programming\\l-ded_ml_models\\files\\df_xpos_images.csv', reduce=None) -> None:
        self.df_reader = DF_Reader(filepath=filepath, reduce=reduce)
        self.df = self.df_reader.df
        self.train_df = self.df_reader.train_df
        self.test_df = self.df_reader.test_df

    def base_model_mapping(self, base_model_str, pooling, input_shape_tuple):
        if base_model_str == 'vgg16':
            base_model = vgg16.VGG16(weights='imagenet', include_top=False, pooling=pooling, input_shape=input_shape_tuple)
            preprocessing_function = vgg16.preprocess_input
        elif base_model_str == 'resnet50':
            base_model = ResNet50(weights='imagenet', include_top=False, pooling=pooling, input_shape=input_shape_tuple)
            preprocessing_function = resnet.preprocess_input
        elif base_model_str == 'resnet101':
            base_model = ResNet101(weights='imagenet', include_top=False, pooling=pooling, input_shape=input_shape_tuple)
            preprocessing_function = resnet.preprocess_input
        elif base_model_str == 'resnet152':
            base_model = ResNet152(weights='imagenet', include_top=False, pooling=pooling, input_shape=input_shape_tuple)
            preprocessing_function = resnet.preprocess_input
        elif base_model_str == 'densenet169':
            base_model = DenseNet169(weights='imagenet', include_top=False, pooling=pooling, input_shape=input_shape_tuple)
            preprocessing_function = densenet.preprocess_input
        elif base_model_str == 'densenet201':
            base_model = DenseNet201(weights='imagenet', include_top=False, pooling=pooling, input_shape=input_shape_tuple)
            preprocessing_function = densenet.preprocess_input
        
        else:
            raise Exception("base model not supported")

        return base_model, preprocessing_function

    def extract_features(self, params, pooling, pipeline) -> pd.DataFrame:
        #Create the model
        base_model, preprocessing_function = self.base_model_mapping(params['base model'], pooling, params['shape'])
        model = self.create_model(base_model)

        #Create the pipeline for extracting features, which just uses all images
        #Create the generators based on the input shape
        pipeline.set_cnn_pipeline(preprocessing_function, self.train_df, self.test_df, params['shape'][:2], params['batch size'], feature_extraction=True)

        #Predict on the cnn with a batch size of 1, grab the output and return it each at a time
        train_features = model.predict(
            pipeline.train_feature_generator,
            batch_size = params['batch size']
        )
        test_features = model.predict(
            pipeline.test_feature_generator,
            batch_size = params['batch size']
        )

        print(f"train_df shape: {self.train_df.shape}, features shape: {train_features.shape}")

        train_features_df = pd.DataFrame(train_features)
        test_features_df = pd.DataFrame(test_features)

        features_1 = pd.concat([self.train_df, train_features_df], axis=1)
        features_2 = pd.concat([self.test_df, test_features_df], axis=1)

        features = pd.concat([features_1, features_2], ignore_index=True)

        return features        

    def create_features_df(self):
        pass

    def create_model(self, base_model) -> Sequential:
        #Make it so the base model layers are untrainable
        base_model.trainable = False

        #Create the new model
        self.model = Sequential()
        self.model.add(base_model)
        
        return self.model

#Adaptive learning rate
def adapt_learning_rate(epoch, lr):
    if epoch < 5:
        return lr
    else:
        #return lr * 0.95
        return lr * tf.math.exp(-0.1)

#R2 calculation for metrics
def r2(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()))

def params_grid_creator(base_models, loss_functions, optimizers_list, learning_rates, input_shapes, epochs_list, batch_sizes):
    grid = {}
    count = 0
    
    #Write the dict for each iteration
    def inner_dict(base_model, loss_function, optimizer, learning_rate, input_shape, epochs, batch_size):
        d = {}
        d['base model'] = base_model
        d['loss function'] = loss_function
        d['optimizer'] = optimizer
        d['learning rate'] =  learning_rate
        d['shape'] = input_shape
        d['epochs'] = epochs
        d['batch size'] = batch_size
        return d
    
    #Loop over each list in the input and create a separate 
    for batch_size in batch_sizes:
        for epochs in epochs_list:
            for input_shape in input_shapes:
                for learning_rate in learning_rates:
                    for optimizer in optimizers_list:
                        for loss_function in loss_functions:
                            for base_model in base_models:
                                grid[str(count)] = inner_dict(base_model, loss_function, optimizer, learning_rate, input_shape, epochs, batch_size)
                                count += 1
                            grid[str(count)] = inner_dict(base_model, loss_function, optimizer, learning_rate, input_shape, epochs, batch_size)
                            count += 1
                        grid[str(count)] = inner_dict(base_model, loss_function, optimizer, learning_rate, input_shape, epochs, batch_size)
                        count += 1
                    grid[str(count)] = inner_dict(base_model, loss_function, optimizer, learning_rate, input_shape, epochs, batch_size)
                    count += 1
                grid[str(count)] = inner_dict(base_model, loss_function, optimizer, learning_rate, input_shape, epochs, batch_size)
                count += 1
            grid[str(count)] = inner_dict(base_model, loss_function, optimizer, learning_rate, input_shape, epochs, batch_size)
            count += 1
        grid[str(count)] = inner_dict(base_model, loss_function, optimizer, learning_rate, input_shape, epochs, batch_size)
        count += 1
    
    #Find a way to compare the function calls
    key_to_remove = set()
    for key in grid.keys():
        try:
            if grid[key] == grid[str(int(key) + 1)]:
                key_to_remove.add(str(int(key) +1))
        except:
            break
    
    for key in key_to_remove:
        remove = grid.pop(key, False)
        if remove == False:
            raise Exception("Key not removed")
            
    #Now add the layers to each value in the dict
    '''for key in grid.keys():
        grid[key]['layers'] = [Flatten(),
            Dense(64, activation='relu'),
            Dense(128, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')]'''
    
    #Display the grid for the user
    print("grid:", pd.DataFrame(grid))
    
    #Calculate number of iterations and ensure the result is in line with the inputs, otherwise raise an error
    number_of_iterations = len(base_models) * len(loss_functions) * len(optimizers_list) * len(learning_rates) * len(input_shapes) * len(epochs_list) * len(batch_sizes)
    if number_of_iterations == len(grid):
        print('number of iterations:', number_of_iterations)
        return grid
    else:
        raise Exception("Mismatched number of iterations")