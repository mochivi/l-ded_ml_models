from helpers import *

def main():
    def test_feature_extractor_svr(params_grid, filepath):
        feature_extractor = Feature_extractor(filepath=filepath, reduce=None)

        for params in params_grid.values():
            print(f"params: {params}")
            features = feature_extractor.extract_features(params=params, pooling='avg', pipeline=Pipeline(data_augmentation=False))
            
            #features.to_csv('features_test.csv')
            
            svr = SVR_model(features_df=features)
            svr.create_train_test_dfs()
            svr.scale_features()
            svr.pca(plot=True)
            svr.run_svr(kernel='rbf', C=3.0, gamma='scale', epsilon=0.05)
            print(svr.score_svr())
            svr.predict_svr(save_csv=True)
            
            break

    def test_reversed_non_reversed_images(params_grid):
        #cnn_no_reverse = CNN_model(filepath='C:\\Users\\victo\\Programming\\l-ded_ml_models\\files\\5deg_4k.csv')
        #cnn_no_reverse.grid_train(params_grid, layers_str, pooling='avg', cnn_pipeline=Pipeline(data_augmentation=False))

        cnn_reversed = CNN_model(filepath='C:\\Users\\victo\\Programming\\l-ded_ml_models\\files\\5deg_4k_reversed.csv')
        cnn_reversed.grid_train(params_grid, layers_str, pooling='avg', cnn_pipeline=Pipeline(data_augmentation=False))

    def new_spline_reg():
        reg_creator = Regression_Creator() #Reads the d_before_spline csv file, which contains xpositions, labels and img paths
        reg_creator.invert_second_line(drop_second_line=True)
        reg_creator.apply_spline_regression(degree=7, knots=8, save=True) #Apply spline regression model and create the new dataframe
        reg_creator.save_new_images() #Checks if images with this degree and knots were already saved and if not, saves them

    def create_df_from_images():
        DF_Creator(dir_path=r'C:\Users\victo\Programming\l-ded_ml_models\images\4degrees_5knots', save_path=r'C:\Users\victo\Programming\l-ded_ml_models\files\7deg_8knots_single_side.csv')

    def run_cnn(params_grid, filepath, pooling, scale_y, width_divide=None, val_split=0.2):
        cnn = CNN_model(filepath=filepath, width_divide=width_divide, scale_y=scale_y)
        cnn_pipeline = Pipeline(data_augmentation=False)
        cnn.grid_train(params_grid, layers_str, pooling=pooling, cnn_pipeline=cnn_pipeline, val_split=val_split)

    #new_spline_reg()
    #create_df_from_images()

    layers_str = [#['flatten'],
        ['dense', 128, 'relu'],
        ['dropout', 0.20],
        ['dense', 256, 'relu'],
        ['dropout', 0.25],
        ['dense', 256, 'relu'],
        ['dropout', 0.20],
        ['dense', 128, 'relu'],
        ['dropout', 0.15],
        ['dense', 64, 'relu'],
        ['dropout', 0.10],
        ['dense', 64, 'relu'],
        ['dropout', 0.10],
        ['dense', 32, 'relu'],
        ['dense', 16, 'relu'],
        ['dense', 16, 'relu'],
        ['dense', 1, 'linear']]
    
    '''layers_str = [['flatten'],
        ['dense', 16, 'relu'],
        ['dropout', 0.25],
        ['dense', 128, 'relu'],
        ['dropout', 0.20],
        ['dense', 64, 'relu'],
        ['dropout', 0.15],
        ['dense', 64, 'relu'],
        ['dropout', 0.15],
        ['dense', 32, 'relu'],
        ['dropout', 0.10],
        ['dense', 32, 'relu'],
        ['dense', 1, 'linear']]'''

    params_grid = params_grid_creator(
        base_models = ['densenet169'],
        loss_functions = ['mae'],
        optimizers_list = ['adam'],
        learning_rates = [0.0001],
        input_shapes = [(170,170,3)],
        epochs_list = [25],
        batch_sizes = [8]
    )

    scale_y=True
    update_global_params(scale_y)

    run_cnn(params_grid,
            filepath=r'C:\Users\victo\Programming\l-ded_ml_models\files\7deg_8knots_single_side.csv',
            pooling='avg',
            scale_y=scale_y,
            width_divide = 1000,
            val_split=0.001)
    
    #test_feature_extractor_svr(params_grid, filepath=r'C:\Users\victo\Programming\l-ded_ml_models\files\7deg_8knots_single_side.csv')
    
if __name__ == '__main__':
    main()