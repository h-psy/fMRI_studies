## 
# This script primarily uses seaborn, a Python data visualization library.
# https://seaborn.pydata.org

import seaborn as sns  
import matplotlib.pyplot as plt
import pandas as pd 
import os
import glob 

# Define directories for input data and output figures
data_dir = ('path to permutation results')  
save_dir = ('path to save figure')  

# Find all CSV files matching the pattern in the specified data directory
files = glob.glob(os.path.join(data_dir, 'your permutation results.csv'))  

# Load the first file in the list of files
num: int = 0  # Specify the index of the file to load
df: pd.DataFrame = pd.read_csv(files[num], sep=',')  # Type hint provided for clarity

# Display the loaded DataFrame in the console for debugging purposes
print(df)

# Set plot resolution and style
plt.rcParams['figure.dpi'] = 600  # Set high DPI for high-resolution figures
sns.set_style('ticks')  # Use the 'ticks' style for a clean appearance

sns.set_color_codes()

# Create a histogram using seaborn
cond = 'NAvsNV', sns.displot(
    df, 
    x='condname, e.g., NAvsNV',
    binwidth=0.035, 
    alpha=1, 
    color=[1, 0.760784313725490, 0.294117647058824]
)

# Construct the output file name
save_file_name = 'figure name you want to save.eps'

# Print save path and file name for debugging purposes
print(save_dir, save_file_name)

# Save the figure to the specified directory
plt.savefig(os.path.join(save_dir, save_file_name), dpi=600, bbox_inches="tight")