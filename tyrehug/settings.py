import os
# Some project settings

DATA_DIR = "/home/charanpal/Data/tyre-hug/"

def get_dir(dir_name):
    if os.path.exists(dir_name):
        return dir_name
    else:
        return ""
