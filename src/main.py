from helpers import *

def main():    
    layers_str = [#['flatten'],
        ['dense', 256, 'relu'],
        ['dropout', 0.25],
        ['dense', 512, 'relu'],
        ['dropout', 0.25],
        ['dense', 128, 'relu'],
        ['dropout', 0.15],
        ['dense', 64, 'relu'],
        ['dropout', 0.10],
        ['dense', 1, 'linear']]

    params_grid = params_grid_creator(base_models = ['vgg16'],
                                    loss_functions = ['mae'],
                                    optimizers_list = ['adam'],
                                    learning_rates = [0.0005],
                                    input_shapes = [(170,170,3)],
                                    epochs_list = [5],
                                    batch_sizes = [32])

    cnn = CNN_model()
    cnn.grid_train(params_grid, layers_str, pooling='avg')

if __name__ == '__main__':
    main()