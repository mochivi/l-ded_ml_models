import os
import pandas as pd

def analyze(many_dirs, poolings, output_path):

    results = {
        'inputs': [],
        'optimizers':[],
        'losses':[],
        'lrs':[],
        'bss':[],
        'maes':[],
        'mapes':[],
        'r2s':[],
        'models': [],
        'folder_refs': [],
        'pooling': []
    }
    for main_dir, pools in zip(many_dirs, poolings):
        for dir in os.listdir(main_dir):
            curr_dir = os.path.join(main_dir,dir)
            for dir_name, subdirs, file_list in os.walk(curr_dir):
                if dir_name.split('\\')[-1][:7] == 'history':
                    results['folder_refs'].append(dir_name)
                    results['pooling'].append(pools)
                for file in file_list:
                    curr_file_path = os.path.join(curr_dir,file)
                    
                    if file == 'params.txt':
                        with open(curr_file_path, 'r', encoding='utf-8') as param_file:
                            for line in param_file.readlines():
                                if line.startswith('input'):
                                    results['inputs'].append(line.split(':')[-1].strip())
                                elif line.startswith('optimizer'):
                                    results['optimizers'].append(line.split(':')[-1].strip())
                                elif line.startswith('loss'):
                                    results['losses'].append(line.split(':')[-1].strip())
                                elif line.startswith('learning'):
                                    results['lrs'].append(line.split(':')[-1].strip())
                                elif line.startswith('batch'):
                                    results['bss'].append(line.split(':')[-1].strip())

                    if file == 'mae_dict.txt':
                        with open(curr_file_path, 'r', encoding='utf-8') as mae_file:
                            for line in mae_file.readlines():
                                if line.startswith('mae'):
                                    results['maes'].append(line.split(':')[-1].strip())
                                elif line.startswith('mape'):
                                    results['mapes'].append(line.split(':')[-1].strip())

                    if file == 'r2_dict.txt':
                        with open(curr_file_path, 'r', encoding='utf-8') as r2_file:
                            for line in r2_file.readlines():
                                if line.startswith('overall'):
                                    results['r2s'].append(line.split(':')[-1].strip())

                    if file == 'modelsummary.txt':
                        with open(curr_file_path, 'r', encoding='utf-8') as model:
                            count = 0
                            for line in model.readlines():
                                if count == 4:
                                    results['models'].append(line.split('(')[0].strip())
                                count += 1
            
    for k, v in results.items():
        print(f"key: {k}, len of list: {len(v)}")                    
    
    df = pd.DataFrame.from_dict(results)
    df.to_csv(output_path, header=True, sep=';', mode='a')
        
if __name__ == "__main__":
    analyze(
        many_dirs = [

        ],
        poolings = [
            
        ],
        output_path = r'D:\Users\Victor\Área de Trabalho\train_history\analyze_results\results.csv'
    )