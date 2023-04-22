

from helpers import *

def main():
    save_path = 'D:\\Users\\Victor\\Área de Trabalho\\train_history\\df\\df_xpos_images.csv'
    read_path = 'C:\\Users\\victo\\Programming\\l-ded_ml_models\\files\\df_xpos_images.csv'
    #df_creator = DF_Creator(save_path=save_path)
    df_reader = DF_Reader(filepath=read_path)

    print(f"train_df shape: {df_reader.train_df.shape}, test_df shape: {df_reader.test_df.shape}")

if __name__ == '__main__':
    main()