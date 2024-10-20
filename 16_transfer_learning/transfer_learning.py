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

[os.listdir('Images/' + i) for i in directories_filtered['directory']]
