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
        base_models = ['resnet152'],
        loss_functions = ['huber_loss'],
        optimizers_list = ['adam'],
        learning_rates = [0.001],
        input_shapes = [(170,170,3)],
        epochs_list = [20],
        batch_sizes = [16]
    )

    cnn = CNN_model()
    cnn.grid_train(params_grid, layers_str, pooling='avg')

if __name__ == '__main__':
    main()