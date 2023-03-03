"""
This file is meant for extracting data from the tensorboard file 
and plotting it in a more visually correct way
"""
import os
# Report only TF errors by default
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import math

# ------- ADJUSTABLES -------
input_paths = [
    # "/mnt/personal/sykorvo1/PPOthesis/ppo/pureEnv/2023-02-24_17-00-49.433548/events.out.tfevents.1677254450.g04.1551953.0.v2",
    # "/mnt/personal/sykorvo1/PPOthesis/ppo/pureEnv/2023-02-24_17-23-48.526468/events.out.tfevents.1677255829.g01.2154405.0.v2",
    # "/mnt/personal/sykorvo1/PPOthesis/ppo/pureEnv/2023-02-27_22-24-03.054114/events.out.tfevents.1677533043.g04.1583837.0.v2",
    # "/mnt/personal/sykorvo1/PPOthesis/ppo/pureEnv/2023-02-27_22-24-03.094545/events.out.tfevents.1677533043.g12.1928932.0.v2",
    # "/mnt/personal/sykorvo1/PPOthesis/ppo/gustyLeft/2023-02-28_18-21-18.481500/events.out.tfevents.1677604878.a01.1440399.0.v2",
    # "/mnt/personal/sykorvo1/PPOthesis/ppo/gustyLeft/2023-02-28_18-21-18.713632/events.out.tfevents.1677604879.a01.1440398.0.v2",
    "/mnt/personal/sykorvo1/PPOthesis/ppo/sides/strength_[0.1, 0.2]/events.out.tfevents.1677708004.a14.510925.0.v2",
    "/mnt/personal/sykorvo1/PPOthesis/ppo/sides/strength_[0.2, 0.3]/events.out.tfevents.1677708004.a14.510926.0.v2",
    "/mnt/personal/sykorvo1/PPOthesis/ppo/sides/strength_[0.3, 0.4]/events.out.tfevents.1677708004.a15.1610646.0.v2",
    "/mnt/personal/sykorvo1/PPOthesis/ppo/sides/strength_[0.4, 0.5]/events.out.tfevents.1677708003.a15.1610647.0.v2",
    "/mnt/personal/sykorvo1/PPOthesis/ppo/BEST/pureEnv/events.out.tfevents.1670670309.a06.1146630.0.v2"
    ]
labels = ["a", "b", "c", "d", "e", "f"]
colors = ['#023eff', '#ff7c00', '#1ac938', '#e8000b', '#8b2be2', '#9f4800', '#f14cc1', '#a3a3a3', '#ffc400', '#00d7ff']
end = 300
legend_curve_label = "PureEnv"
xlabel = 'steps'
ylabel = 'Average Score'
save_file = "plot.png"
# ---------------------------
plt.style.use('seaborn-v0_8')
""" seaborn colors https://seaborn.pydata.org/tutorial/color_palettes.html
STYLE = "bright"
print(sns.color_palette(STYLE).as_hex())
sns.color_palette(STYLE).as_hex()
"""

CB91_Purple = '#9D2EC5'
CB91_Blue = '#2CBDFE'
CB91_Green = '#47DBCD'
CB91_Pink = '#F3A0F2'
CB91_Violet = '#661D98'
CB91_Amber = '#F5B14C'
color_list = [CB91_Purple, CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber, CB91_Violet]
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)
#  --------------------------

def get_name_of_last_dir(path):
    return os.path.basename(os.path.dirname(path)) # get the name of the last directory

def get_data_from_tbfile(file):
    # Load Tensorboard data
    data = tf.compat.v1.train.summary_iterator(file)

    # extract out of the tensorboard file only what we want
    steps = []
    values = []
    for e in data:
        for v in e.summary.value:
            if v.tag == 'average score':
                # print(tf.make_ndarray(v.tensor))
                steps.append(e.step)
                values.append(float(tf.make_ndarray(v.tensor)))
                # print(f"{e=}\n{e.step=}\n{e.summary.value=}")
                # exit()

    # print(f"{steps=}\n\n{values=}\n\n{error=}")
    # print(f"{len(steps)=}")
    return steps, values

def make_one_graph(tb_input, ax, output_file = None):
    steps, values = get_data_from_tbfile(tb_input)

    # Plot values
    ax.plot(steps[:end], values[:end], label=legend_curve_label)

    # https://stackoverflow.com/questions/12957582/plot-yerr-xerr-as-shaded-region-rather-than-error-bars
    # error = np.random.normal(1, 20, size=len(values))
    # ax.fill_between(steps, values-error, values+error)

    # ax.title('My title')
    ax.xlabel(xlabel, fontweight="bold")
    ax.ylabel(ylabel, fontweight="bold")
    legend_properties = {'weight':'bold'}
    ax.legend(loc = "best", prop = legend_properties)

    # ax.show()
    if output_file is not None:
        ax.savefig(output_file)
        print(f"Saving {output_file}")

# def make_all_graphs(input_paths):
#     figs = []
#     for path in input_paths:
#         save_file_name = os.path.basename(os.path.dirname(path)) # get the name of the last directory
#         figs.append(make_one_graph(path, f"{save_file_name}.png"))
#     return figs

def figs_in_line(input_paths, labels, colors):
    fig, axs = plt.subplots(ncols=len(input_paths), nrows=1)

    for i, path in enumerate(input_paths):
        s, v = get_data_from_tbfile(path)
        s = np.array(s) / 1000
        axs[i].plot(s[:300], v[:300], label = labels[i], color = colors[i])
        axs[i].set_title(f"{colors[i]}")
        # axs[i].ticklabel_format(useMathText=True, style='sci', axis='x', scilimits=(1,4))
        
    fig.supxlabel('steps ($10^3$)')
    fig.supylabel('average score')
    
    return fig, axs

STEPS_POWER = 6 # xlabel 10^power and also xaxis / 10^power
STEP_CUTOFF = 700
def more_lines_in_one_graph(input_paths, ax, colors, title, labels = None):
    for i, path in enumerate(input_paths):
        s, v = get_data_from_tbfile(path)
        s = np.array(s) / math.pow(10, STEPS_POWER)
        ax.plot(s[:STEP_CUTOFF], v[:STEP_CUTOFF], label = get_name_of_last_dir(path), color = colors[i])
    ax.set_title(title)
    ax.legend()

def get_sides(ax, colors):
    input_paths = [
        "/mnt/personal/sykorvo1/PPOthesis/ppo/sides/strength_[0.1, 0.2]/events.out.tfevents.1677708004.a14.510925.0.v2",
        "/mnt/personal/sykorvo1/PPOthesis/ppo/sides/strength_[0.2, 0.3]/events.out.tfevents.1677708004.a14.510926.0.v2",
        "/mnt/personal/sykorvo1/PPOthesis/ppo/sides/strength_[0.3, 0.4]/events.out.tfevents.1677708004.a15.1610646.0.v2",
        "/mnt/personal/sykorvo1/PPOthesis/ppo/sides/strength_[0.4, 0.5]/events.out.tfevents.1677708003.a15.1610647.0.v2",
        "/mnt/personal/sykorvo1/PPOthesis/ppo/BEST/pureEnv/events.out.tfevents.1670670309.a06.1146630.0.v2"
    ]
    
    more_lines_in_one_graph(input_paths, ax, colors, "Continuous wind from both sides")

def get_right(ax, colors):
    input_paths = [
        "/mnt/personal/sykorvo1/PPOthesis/ppo/right/strength_[0.1, 0.2]/events.out.tfevents.1677749546.a10.1203167.0.v2",
        "/mnt/personal/sykorvo1/PPOthesis/ppo/right/strength_[0.2, 0.3]/events.out.tfevents.1677749546.a10.1203168.0.v2",
        "/mnt/personal/sykorvo1/PPOthesis/ppo/right/strength_[0.3, 0.4]/events.out.tfevents.1677749546.a11.2519113.0.v2",
        "/mnt/personal/sykorvo1/PPOthesis/ppo/right/strength_[0.4, 0.5]/events.out.tfevents.1677749547.a11.2519114.0.v2",
        "/mnt/personal/sykorvo1/PPOthesis/ppo/BEST/pureEnv/events.out.tfevents.1670670309.a06.1146630.0.v2"
    ]
    
    more_lines_in_one_graph(input_paths, ax, colors, "Continuous Right wind")
    
def get_gustyRight(ax, colors):
    input_paths = [
        "/mnt/personal/sykorvo1/PPOthesis/ppo/gustyRight/strength_[0.1, 0.2]/events.out.tfevents.1677749527.a10.1202568.0.v2",
        "/mnt/personal/sykorvo1/PPOthesis/ppo/gustyRight/strength_[0.2, 0.3]/events.out.tfevents.1677749528.a10.1202569.0.v2",
        "/mnt/personal/sykorvo1/PPOthesis/ppo/gustyRight/strength_[0.3, 0.4]/events.out.tfevents.1677749528.a11.2518530.0.v2",
        "/mnt/personal/sykorvo1/PPOthesis/ppo/gustyRight/strength_[0.4, 0.5]/events.out.tfevents.1677749528.a11.2518531.0.v2",
        "/mnt/personal/sykorvo1/PPOthesis/ppo/BEST/pureEnv/events.out.tfevents.1670670309.a06.1146630.0.v2"
    ]
    
    more_lines_in_one_graph(input_paths, ax, colors, "Gusty Right wind")
    
def get_left(ax, colors):
    input_paths = [
        "/mnt/personal/sykorvo1/PPOthesis/ppo/left/strength_[0.1, 0.2]/events.out.tfevents.1677749342.a10.1201228.0.v2",
        "/mnt/personal/sykorvo1/PPOthesis/ppo/left/strength_[0.2, 0.3]/events.out.tfevents.1677790045.a10.1229252.0.v2",
        "/mnt/personal/sykorvo1/PPOthesis/ppo/left/strength_[0.3, 0.4]/events.out.tfevents.1677749342.a11.2517219.0.v2",
        "/mnt/personal/sykorvo1/PPOthesis/ppo/left/strength_[0.4, 0.5]/events.out.tfevents.1677749342.a11.2517220.0.v2",
        "/mnt/personal/sykorvo1/PPOthesis/ppo/BEST/pureEnv/events.out.tfevents.1670670309.a06.1146630.0.v2"
    ]
    
    more_lines_in_one_graph(input_paths, ax, colors, "Continuous Left wind")
    
def get_gustyLeft(ax, colors):
    input_paths = [
        "/mnt/personal/sykorvo1/PPOthesis/ppo/gustyLeft/strength_[0.1, 0.2]/events.out.tfevents.1677749452.a10.1201906.0.v2",
        "/mnt/personal/sykorvo1/PPOthesis/ppo/gustyLeft/strength_[0.2, 0.3]/events.out.tfevents.1677749452.a10.1201907.0.v2",
        "/mnt/personal/sykorvo1/PPOthesis/ppo/gustyLeft/strength_[0.3, 0.4]/events.out.tfevents.1677749451.a11.2517891.0.v2",
        "/mnt/personal/sykorvo1/PPOthesis/ppo/gustyLeft/strength_[0.4, 0.5]/events.out.tfevents.1677749452.a11.2517892.0.v2",
        "/mnt/personal/sykorvo1/PPOthesis/ppo/BEST/pureEnv/events.out.tfevents.1670670309.a06.1146630.0.v2"
    ]
    
    more_lines_in_one_graph(input_paths, ax, colors, "Gusty Left wind")

if __name__ == '__main__':
    # fig, axs = figs_in_line(input_paths, labels, colors)
    # plt.tight_layout()
    # fig.legend(loc="center right")
    # fig.savefig("a.png")
    
    # make_one_graph(input_paths[0], axs[0], output_file = None)
    
    fig, axs = plt.subplots(ncols=1, nrows=1)
    
    # get_sides(axs[0], colors)
    # get_right(axs, colors)
    # get_gustyRight(axs, colors)
    get_left(axs, colors)
    # get_gustyLeft(axs, colors)
    
    fig.supxlabel(f'steps ($10^{STEPS_POWER}$)')
    fig.supylabel('average score')
    fig.savefig("left.png")
    
    

    
"""
https://stackoverflow.com/questions/42656139/set-scientific-notation-with-fixed-exponent-and-significant-digits-for-multiple
"""