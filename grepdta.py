import os
import shutil
# Set the root directory
root_dir = './multirun/2023-02-27/'

# Loop through all subdirectories and files
log_file = './profilelogs/all_logs.csv'

with open(log_file, 'w') as f:
    f.write("-,-,random , #gpu  , nanobatch, layer, micro_reuse_rate[layer], full_reuse_rate[layer] \n")
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            # Check if the file has a .log extension
            if file.endswith('.log'):
                with open(subdir+"/"+file, 'r') as log:
                    f.write(log.read())


shutil.rmtree(root_dir)