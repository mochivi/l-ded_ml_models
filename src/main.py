from helpers import *

def main():    
    layers_str = [#['flatten'],
        ['dense', 256, 'relu'],
        ['dropout', 0.25],
        ['dense', 512, 'relu'],
        ['dropout', 0.20],
        ['dense', 256, 'relu'],
        ['dropout', 0.15],
        ['dense', 256, 'relu'],
        ['dropout', 0.10],
        ['dense', 128, 'relu'],
        ['dropout', 0.10],
        ['dense', 64, 'relu'],
        ['dropout', 0.10],
        ['dense', 32, 'relu'],
        ['dropout', 0.10],
        ['dense', 16, 'relu'],
        ['dropout', 0.10],
        ['dense', 16, 'relu'],
        ['dropout', 0.10],
        ['dense', 1, 'linear']]

    params_grid = params_grid_creator(
        base_models = ['densenet201', 'resnet152'],
        loss_functions = ['mae'],
        optimizers_list = ['adam'],
        learning_rates = [0.0001],
        input_shapes = [(170,170,3)],
        epochs_list = [50],
        batch_sizes = [8]
    )

    def test_feature_extractor_svr(params_grid):
        feature_extractor = Feature_extractor(reduce=None)
        features = feature_extractor.extract_features(params=params, pooling='avg', pipeline=Pipeline(data_augmentation=False))
        
        for params in params_grid.values():
            print(f"params: {params}")
            
            #features.to_csv('features_test.csv')
            
            svr = SVR_model(features_df=features)
            svr.create_train_test_dfs()
            svr.scale_features()
            svr.pca(plot=True)
            svr.run_svr(kernel='rbf', C=3.0, gamma='scale', epsilon=0.05)
            #print(svr.score_svr())
            svr.predict_svr()       
            
            break

    def run_cnn(params_grid):
        cnn = CNN_model()
        cnn_pipeline = Pipeline(data_augmentation=False)
        cnn.grid_train(params_grid, layers_str, pooling=None, cnn_pipeline=cnn_pipeline)

    def new_spline_reg():
        reg_creator = Regression_Creator() #Reads the d_before_spline csv file, which contains xpositions, labels and img paths
        #reg_creator.invert_second_line()
        reg_creator.apply_spline_regression(degree=5, knots=4, save=True) #Apply spline regression model and create the new dataframe
        reg_creator.save_new_images() #Checks if images with this degree and knots were already saved and if not, saves them

    def test_reversed_non_reversed_images(params_grid):
        cnn_no_reverse = CNN_model(filepath='C:\\Users\\victo\\Programming\\l-ded_ml_models\\files\\5deg_4k.csv')
        cnn_no_reverse.grid_train(params_grid, layers_str, pooling='avg', cnn_pipeline=Pipeline(data_augmentation=False))

        #cnn_reversed = CNN_model(filepath='C:\\Users\\victo\\Programming\\l-ded_ml_models\\files\\5deg_4k_reversed.csv')
        #cnn_reversed.grid_train(params_grid, layers_str, pooling='avg', cnn_pipeline=Pipeline(data_augmentation=False))

    test_reversed_non_reversed_images(params_grid)
    #run_cnn(params_grid)
    #test_feature_extractor_svr(params_grid)
    
if __name__ == '__main__':
    main()