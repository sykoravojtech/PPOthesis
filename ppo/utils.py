import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
import json
import tensorflow as tf
from time import sleep
import random
from my_parser import *

# colors for printing in the terminal
class bcolors:
    PURPLE = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    ITALICIZE = '\033[3m'
    UNDERLINE = '\033[4m'
    PURPLE_BACK = '\033[105m'
    
def show_terminal_colors():
    print(f"{bcolors.PURPLE} Header {bcolors.ENDC}")
    print(f"{bcolors.PURPLE_BACK} Header info {bcolors.ENDC}")
    print(f"{bcolors.BLUE} blue {bcolors.ENDC}")
    print(f"{bcolors.CYAN} cyan {bcolors.ENDC}")
    print(f"{bcolors.GREEN} green {bcolors.ENDC}")
    print(f"{bcolors.YELLOW} warning {bcolors.ENDC}")
    print(f"{bcolors.RED} fail {bcolors.ENDC}")
    print(f"{bcolors.BOLD} bold {bcolors.ENDC}")
    print(f"{bcolors.ITALICIZE} italicize {bcolors.ENDC}")
    print(f"{bcolors.UNDERLINE} underline {bcolors.ENDC}")

def get_logger(model_dir, tensorboard=False):
    """
    tensorboard --logdir='/your/path/here'
    """
    if tensorboard:
        tensorboard_log_dir = model_dir if model_dir[-1]=="/" else model_dir + "/"
        summary_writer = tf.summary.create_file_writer(tensorboard_log_dir)
        return summary_writer
    return None

def print_info(env, args):
    print(f"==== {bcolors.PURPLE}ENV INFO{bcolors.ENDC} ====")
    print(f"{bcolors.PURPLE_BACK}{env.action_space=}\n{env.observation_space=}{bcolors.ENDC}")
    print("==================")
    print(f"====== {bcolors.PURPLE}ARGS{bcolors.ENDC} ======")
    print(f"{bcolors.PURPLE_BACK}{args.__dict__}{bcolors.ENDC}")
    print("==================")

def get_curr_datetime(microsecond=False):
    if microsecond:
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
def add_curr_time_to_dir(dir, microsecond=False):
    return os.path.join(dir, get_curr_datetime(microsecond))

def create_dir_for_curr_runtime(models_dir):
    """adds a subfolder in the models directory for this runtime

    Args:
        models_dir (string): path to directory with models

    Returns:
        string: models_dir with a new current datetime subfolder
    """
    sleep(random.random()) # so that the parallel executions dont save into the same folder
    models_dir = add_curr_time_to_dir(models_dir, microsecond=True)
    os.makedirs(models_dir)
    print_notification_style(f"This runtime saved in {models_dir}")
    return models_dir

def save_pltgraph(avg_score_history, chkpt_dir, e, start_from_ep):
    fig_name = os.path.join(chkpt_dir, "mean_returns")
    
    plt.plot(
        np.linspace(start_from_ep, e, len(avg_score_history)), 
        avg_score_history)
    plt.title(f"Average Return for last 300 episodes")
    plt.xlabel("episode")
    plt.ylabel("average return")
    plt.savefig(f"{fig_name}.png")
    plt.clf()
    
def save_model_summary(model_dir, model):
    filename = os.path.join(model_dir, "model_summary.txt")
    with open(filename,'w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
        model.get_layer('CNN_model').summary(print_fn=lambda x: fh.write(x + '\n'))
        # model.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))

def create_subfolder(models_dir, addon=""):
    sleep(random.random()) # so that the parallel executions dont save into the same folder
    models_dir = os.path.join(models_dir, addon)
    os.makedirs(models_dir)
    print_notification_style(f"This runtime saved in {models_dir}")
    return models_dir

def print_chapter_style(text):
    print("=" * (len(text) + 8))
    print(f">>> {bcolors.PURPLE}{text}{bcolors.ENDC} <<<")
    print("=" * (len(text) + 8))
    
def print_notification_style(text):
    print(f">> {bcolors.CYAN}{text}{bcolors.ENDC}")
    
def print_divider(divider_length = 30):
    print(f"{bcolors.YELLOW}{'-' * divider_length}{bcolors.ENDC}")
    
def print_tensorflow_version():
    import tensorflow as tf
    print_notification_style(f"TensorFlow version = {tf.__version__}")
    
def print_available_devices():
    from tensorflow.python.client import device_lib
    print_notification_style(f"Devices available: {[device.name for device in device_lib.list_local_devices() if device.name != None]}")   

def get_name_of_last_dir(path):
    return os.path.basename(os.path.dirname(path)) # get the name of the last directory

def round_list(lst, decimals=2):
    return [round(x, decimals) for x in lst]