from helpers import *



def main():    
    layers_str = [#['flatten'],
        ['dense', 256, 'relu'],
        #['dropout', 0.25],
        ['dense', 512, 'relu'],
        #['dropout', 0.25],
        ['dense', 128, 'relu'],
        #['dropout', 0.15],
        ['dense', 64, 'relu'],
        #['dropout', 0.10],
        ['dense', 1, 'linear']]

    params_grid = params_grid_creator(
        base_models = ['densenet169'],
        loss_functions = ['huber_loss'],
        optimizers_list = ['adam'],
        learning_rates = [0.001],
        input_shapes = [(170,170,3)],
        epochs_list = [20],
        batch_sizes = [8]
    )

    def test_feature_extractor():
        feature_extractor = Feature_extractor(reduce=None)
        for params in params_grid.values():
            print(f"params: {params}")
            features = feature_extractor.extract_features(params=params, pooling='avg', pipeline=Pipeline(data_augmentation=False))
            #features.to_csv('features_test.csv')
            
            svr = SVR_model(features_df=features)
            svr.create_train_test_dfs()
            svr.scale_features()
            svr.pca(plot=True)
            svr.run_svr(kernel='rbf', C=3.0, gamma='scale', epsilon=0.05)
            #print(svr.score_svr())
            svr.predict_svr()       
            
            break

    def run_cnn():
        cnn = CNN_model()
        cnn_pipeline = Pipeline(data_augmentation=False)
        cnn.grid_train(params_grid, layers_str, pooling='avg', pipeline=cnn_pipeline)

    #run_cnn()
    test_feature_extractor()
    

if __name__ == '__main__':
    main()