import os
import shutil

root_dir = "./multirun/2023-04-23/"
# root_dir = "./outputs/2023-04-07/"
log_file = "./profilelogs/all_logs.csv"

with open(log_file, "w") as f:
    f.write(
        "-,- , model,dataset_name  , nanobatch, layer, mean_epoch_time(s), batch size,max GPU mem(G)\n"
    )
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".log"):
                with open(subdir + "/" + file, "r") as log:
                    f.write(log.read())

shutil.rmtree(root_dir)
