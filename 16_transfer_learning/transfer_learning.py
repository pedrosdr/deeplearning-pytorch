import os
import shutil
import numpy as np
import pandas as pd
import seaborn as sns


directories_info = pd.DataFrame({
    'directory': [dirs for root, dirs, files in os.walk('Images')][0],
    'number_of_files': [len(files) for root, dirs, files in os.walk('Images')][1:]  
})

directories_info = directories_info.sort_values('number_of_files', axis=0)

sns.lineplot(
    data=directories_info, 
    x=range(len(directories_info)), 
    y='number_of_files'
)

directories_filtered = directories_info[
    directories_info['number_of_files'] >= 600    
]

directories_to_remove = [x for x in directories_info[
    ~directories_info['directory'].isin(directories_filtered['directory'])
   ]['directory']
]

[shutil.rmtree('Images/' + directory) for directory in directories_to_remove]

numfiles_per_folder = [len(files) for root, dirs, files in os.walk('Images') if len(files) > 0]
filenames = [os.listdir('Images/' + i) for i in directories_filtered['directory']]

np.random.choice(filenames[0], 30, replace=False)
np.random.seed(23)
files_to_test = [sorted(np.random.choice(files, round(600 * 0.2), replace=False)) 
                 for files, num_files in zip(
                             filenames, numfiles_per_folder
                         )]

files_to_test[0]
