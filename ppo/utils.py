import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from time import sleep
import random
from gym.vector.vector_env import VectorEnv

from my_parser import *


class bcolors:
    """ 
    Class for printing colorful text in the terminal
    """
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
    
    
def show_terminal_colors() -> None:
    """
    Print into the terminal all our colors to showcase them
    """
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


def get_logger(model_dir: str, tensorboard: bool = False): # -> tf.summary.SummaryWriter
    """ Create a tensorboard summary writer
    Command for opening the summary file
        tensorboard --logdir='/your/path/here'
        
    Args:
        model_dir: path to the directory where we want the summary to be saved
        tensorboard: True if we want to make the summary
    """
    if tensorboard:
        tensorboard_log_dir = model_dir if model_dir[-1]=="/" else model_dir + "/"
        summary_writer = tf.summary.create_file_writer(tensorboard_log_dir)
        return summary_writer
    return None


def print_info(env: VectorEnv, args: argparse.Namespace) -> None:
    """ Print information about the environment and print all arguments from our parser

    Args:
        env: environment
        args: arguments
    """
    print(f"==== {bcolors.PURPLE}ENV INFO{bcolors.ENDC} ====")
    print(f"{bcolors.PURPLE_BACK}{env.action_space=}\n{env.observation_space=}{bcolors.ENDC}")
    print("==================")
    print(f"====== {bcolors.PURPLE}ARGS{bcolors.ENDC} ======")
    print(f"{bcolors.PURPLE_BACK}{args.__dict__}{bcolors.ENDC}")
    print("==================")


def get_curr_datetime(microsecond: bool = False) -> str:
    """ Returns the current time as a string
    
    Args:
        microsecond: True if we want to add the microsecond number
    """
    if microsecond:
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    
def add_curr_time_to_dir(dir: str, microsecond:bool = False) -> str:
    """ Add a new directory named as the current time. 
    This is done to be able to start multiple trainings at the same time and them having separate folders to save weights in.

    Args:
        dir: original directory
        microsecond: If True add the microsecond number to the directory name
    """
    return os.path.join(dir, get_curr_datetime(microsecond))


def create_dir_for_curr_runtime(models_dir: str) -> str:
    """ Adds a subfolder in the models directory for this runtime

    Args:
        models_dir: path to directory with models
    """
    sleep(random.random()) # so that the parallel executions dont save into the same folder
    models_dir: str = add_curr_time_to_dir(models_dir, microsecond=True)
    os.makedirs(models_dir)
    print_notification_style(f"This runtime saved in {models_dir}")
    return models_dir


def save_pltgraph(avg_score_history: list[float], chkpt_dir: str, e: int, start_from_ep: int) -> None:
    """ Save an image of a plot showing the average score history

    Args:
        avg_score_history: list of average scores
        chkpt_dir: path to the directory to save the image in
        e: episode number
        start_from_ep: the episode to start on
    """
    fig_name: str = os.path.join(chkpt_dir, "mean_returns")
    
    plt.plot(
        np.linspace(start_from_ep, e, len(avg_score_history)), 
        avg_score_history)
    plt.title(f"Average Return for last 300 episodes")
    plt.xlabel("episode")
    plt.ylabel("average return")
    plt.savefig(f"{fig_name}.png")
    plt.clf()
    
    
def save_model_summary(model_dir: str, model: Model) -> None:
    """ Save the keras.Model summary
    
    Args:
        model_dir: path to the directory in which to save the model summary
        model: the Model
    """
    filename: str = os.path.join(model_dir, "model_summary.txt")
    with open(filename,'w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
        model.get_layer('CNN_model').summary(print_fn=lambda x: fh.write(x + '\n'))
        # model.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))


def create_subfolder(models_dir: str, addon: str = "") -> str:
    """ Create a new folder inside the given directory

    Args:
        models_dir: the given directory
        addon (str, optional): if we want to add some text at the end of the directory name
    """
    sleep(random.random()) # so that the parallel executions dont save into the same folder
    models_dir = os.path.join(models_dir, addon)
    os.makedirs(models_dir)
    print_notification_style(f"This runtime saved in {models_dir}")
    return models_dir


def print_chapter_style(text: str) -> None:
    """ 
    Wrap the text in some fancy style and purple color and print it out
    """
    print("=" * (len(text) + 8))
    print(f">>> {bcolors.PURPLE}{text}{bcolors.ENDC} <<<")
    print("=" * (len(text) + 8))
    
    
def print_notification_style(text: str) -> None:
    """ 
    Wrap the text in cyan color and print it out
    """
    print(f">> {bcolors.CYAN}{text}{bcolors.ENDC}")
    
    
def print_divider(divider_length: int = 30) -> None:
    """ 
    Print a yellow line to act as a divider
    """
    print(f"{bcolors.YELLOW}{'-' * divider_length}{bcolors.ENDC}")
    
    
def print_tensorflow_version():
    import tensorflow as tf
    print_notification_style(f"TensorFlow version = {tf.__version__}")
    
    
def print_available_devices():
    """ 
    Print which CPUs and GPUs are visible to TensorFlow
    """
    from tensorflow.python.client import device_lib
    print_notification_style(f"Devices available: {[device.name for device in device_lib.list_local_devices() if device.name != None]}")   


def get_name_of_last_dir(path: str) -> str:
    """
    Returns the name of the last directory in the path
    """
    return os.path.basename(os.path.dirname(path))


def round_list(lst: list[float], decimals: int = 2) -> list[float]:
    """
    Round all of the numbers in the list to the given number of decimal places
    """
    return [round(x, decimals) for x in lst]


def add_col_to_start(rows: list[str | float], col: list[str]) -> list[str | float]:
    """Add the column with model names to the start of the rows matrix
    It is used for saving the csv files of running model tests.
    
    Args:
        rows: 2d array of information
        col: the column to put at the left of the rows matrix
    """
    assert len(rows) == len(col)
    new_rows = [row[:] for row in rows]
    for i in range(len(col)):
        new_rows[i].insert(0, col[i])
    return new_rows


if __name__ == "__main__":
    print()
    print(add_col_to_start(rows = [[1,2,3],[4,5,6]], col = ["a", "b"]))