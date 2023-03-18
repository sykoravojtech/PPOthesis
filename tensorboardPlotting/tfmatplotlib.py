"""
This file is meant for extracting data from the tensorboard file 
and plotting it in a more visually correct way
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import math
from typing import List, Tuple, Any

# ------- ADJUSTABLES -------
labels = ["a", "b", "c", "d", "e", "f"]
colors = ['#023eff', '#ff7c00', '#1ac938', '#e8000b', '#8b2be2', '#9f4800', '#f14cc1', '#a3a3a3', '#ffc400', '#00d7ff']
end = 300
legend_curve_label = "noWind"
xlabel = 'steps'
ylabel = 'Average Score'
save_file = "plot.png"
# ---------------------------
plt.style.use('seaborn-v0_8')
""" 
seaborn colors https://seaborn.pydata.org/tutorial/color_palettes.html
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

def get_name_of_last_dir(path: str) -> str:
    """ Reutrns the name of the Last directory in the path

    Args:
        path: directory name
    """
    return os.path.basename(os.path.dirname(path)) # get the name of the last directory

def get_data_from_tbfile(file: str) -> Tuple[List[int], List[Any]]:
    """ Load data from a TensorBoard file

    Args:
        file: path to the TensorBoard file  
    """
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

def make_one_graph(tb_input: str, ax: plt.Axes, output_file = None):
    """ Plot a curve using tensorboard data

    Args:
        tb_input: TensorBoard file path
        ax: matplotlib axes to plot on
        output_file:
    """
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

def figs_in_line(input_paths: List[str], labels: List[str], colors: List[str]) -> Tuple[plt.Figure, plt.Axes]:
    """ Plot all inputs as separate plots in a single line

    Args:
        input_paths (_type_): _description_
        labels (_type_): _description_
        colors (_type_): _description_
    """
    fig, axs = plt.subplots(ncols=len(input_paths), nrows=1)

    # for each input path, plot it on a separate plot
    for i, path in enumerate(input_paths):
        s, v = get_data_from_tbfile(path)
        s = np.array(s) / 1000
        axs[i].plot(s[:300], v[:300], label = labels[i], color = colors[i])
        axs[i].set_title(f"{colors[i]}")
        # axs[i].ticklabel_format(useMathText=True, style='sci', axis='x', scilimits=(1,4))
    
    # change the global axes labels
    fig.supxlabel('steps ($10^3$)')
    fig.supylabel('average score')
    
    return fig, axs

STEPS_POWER: int = 6 # xlabel 10^power and also xaxis / 10^power
def more_lines_in_one_graph(input_paths: List[str], ax: plt.Axes, colors: List[str], title: str, labels: List[str] = None, step_cutoff: int = 550) -> None:
    """ Plot more curves in a single plot

    Args:
        input_paths : list of paths to tensorboard files of each curve
        ax : matplotlib axes to plot on
        colors : list of colors to use
        title : title displayed on the plot
        labels : list of labels corresponding to each path
        step_cutoff : tensorboard data can be 1500 steps long so this only takes a part for the graph
    """
    
    for i, path in enumerate(input_paths):
        s, v = get_data_from_tbfile(path)
        s = np.array(s) / math.pow(10, STEPS_POWER)
        if labels is None:
            ax.plot(s[:step_cutoff], v[:step_cutoff], label = get_name_of_last_dir(path), color = colors[i])
        else:
            ax.plot(s[:step_cutoff], v[:step_cutoff], label = labels[i], color = colors[i])
    ax.set_title(title)
    ax.legend()

# ---------------- Graphing a curve of the pretrained models -----------------

def get_left(ax: plt.Axes, colors: List[str]) -> None:
    input_paths = [
        "/mnt/personal/sykorvo1/PPOthesis/ppo/left/strength_[0.1, 0.2]/events.out.tfevents.1677749342.a10.1201228.0.v2",
        "/mnt/personal/sykorvo1/PPOthesis/ppo/left/strength_[0.2, 0.3]/events.out.tfevents.1677790045.a10.1229252.0.v2",
        "/mnt/personal/sykorvo1/PPOthesis/ppo/left/strength_[0.3, 0.4]/events.out.tfevents.1677749342.a11.2517219.0.v2",
        "/mnt/personal/sykorvo1/PPOthesis/ppo/left/strength_[0.4, 0.5]/events.out.tfevents.1677749342.a11.2517220.0.v2",
        "/mnt/personal/sykorvo1/PPOthesis/ppo/BEST/noWind/events.out.tfevents.1670670309.a06.1146630.0.v2"
    ]
    
    more_lines_in_one_graph(input_paths, ax, colors, "Continuous Left wind", step_cutoff = 550)

def get_right(ax: plt.Axes, colors: List[str]) -> None:
    input_paths = [
        "/mnt/personal/sykorvo1/PPOthesis/ppo/right/strength_[0.1, 0.2]/events.out.tfevents.1677749546.a10.1203167.0.v2",
        "/mnt/personal/sykorvo1/PPOthesis/ppo/right/strength_[0.2, 0.3]/events.out.tfevents.1677749546.a10.1203168.0.v2",
        "/mnt/personal/sykorvo1/PPOthesis/ppo/right/strength_[0.3, 0.4]/events.out.tfevents.1677749546.a11.2519113.0.v2",
        "/mnt/personal/sykorvo1/PPOthesis/ppo/right/strength_[0.4, 0.5]/events.out.tfevents.1677749547.a11.2519114.0.v2",
        "/mnt/personal/sykorvo1/PPOthesis/ppo/BEST/noWind/events.out.tfevents.1670670309.a06.1146630.0.v2"
    ]
    
    more_lines_in_one_graph(input_paths, ax, colors, "Continuous Right wind", step_cutoff = 550)
    
def get_gustyLeft(ax: plt.Axes, colors: List[str]) -> None:
    input_paths = [
        "/mnt/personal/sykorvo1/PPOthesis/ppo/gustyLeft/strength_[0.1, 0.2]/events.out.tfevents.1677749452.a10.1201906.0.v2",
        "/mnt/personal/sykorvo1/PPOthesis/ppo/gustyLeft/strength_[0.2, 0.3]/events.out.tfevents.1677749452.a10.1201907.0.v2",
        "/mnt/personal/sykorvo1/PPOthesis/ppo/gustyLeft/strength_[0.3, 0.4]/events.out.tfevents.1677749451.a11.2517891.0.v2",
        "/mnt/personal/sykorvo1/PPOthesis/ppo/gustyLeft/strength_[0.4, 0.5]/events.out.tfevents.1677749452.a11.2517892.0.v2",
        "/mnt/personal/sykorvo1/PPOthesis/ppo/BEST/noWind/events.out.tfevents.1670670309.a06.1146630.0.v2"
    ]
    
    more_lines_in_one_graph(input_paths, ax, colors, "Gusty Left wind", step_cutoff = 700)
    
def get_gustyRight(ax: plt.Axes, colors: List[str]) -> None:
    input_paths = [
        "/mnt/personal/sykorvo1/PPOthesis/ppo/gustyRight/strength_[0.1, 0.2]/events.out.tfevents.1677749527.a10.1202568.0.v2",
        "/mnt/personal/sykorvo1/PPOthesis/ppo/gustyRight/strength_[0.2, 0.3]/events.out.tfevents.1677749528.a10.1202569.0.v2",
        "/mnt/personal/sykorvo1/PPOthesis/ppo/gustyRight/strength_[0.3, 0.4]/events.out.tfevents.1677749528.a11.2518530.0.v2",
        "/mnt/personal/sykorvo1/PPOthesis/ppo/gustyRight/strength_[0.4, 0.5]/events.out.tfevents.1677749528.a11.2518531.0.v2",
        "/mnt/personal/sykorvo1/PPOthesis/ppo/BEST/noWind/events.out.tfevents.1670670309.a06.1146630.0.v2"
    ]
    
    more_lines_in_one_graph(input_paths, ax, colors, "Gusty Right wind", step_cutoff = 450)

def get_sides(ax: plt.Axes, colors: List[str]) -> None:
    input_paths = [
        "/mnt/personal/sykorvo1/PPOthesis/ppo/sides/strength_[0.1, 0.2]/events.out.tfevents.1677708004.a14.510925.0.v2",
        "/mnt/personal/sykorvo1/PPOthesis/ppo/sides/strength_[0.2, 0.3]/events.out.tfevents.1677708004.a14.510926.0.v2",
        "/mnt/personal/sykorvo1/PPOthesis/ppo/sides/strength_[0.3, 0.4]/events.out.tfevents.1677708004.a15.1610646.0.v2",
        "/mnt/personal/sykorvo1/PPOthesis/ppo/sides/strength_[0.4, 0.5]/events.out.tfevents.1677708003.a15.1610647.0.v2",
        "/mnt/personal/sykorvo1/PPOthesis/ppo/BEST/noWind/events.out.tfevents.1670670309.a06.1146630.0.v2"
    ]
    
    more_lines_in_one_graph(input_paths, ax, colors, "Continuous wind from both sides", step_cutoff = 450)

def get_gustySides(ax: plt.Axes, colors: List[str]) -> None:
    input_paths = [
        "/mnt/personal/sykorvo1/PPOthesis/ppo/gustySides/strength_[0.1, 0.2]/events.out.tfevents.1677864061.a10.1274552.0.v2",
        "/mnt/personal/sykorvo1/PPOthesis/ppo/gustySides/strength_[0.2, 0.3]/events.out.tfevents.1677864060.a10.1274553.0.v2",
        "/mnt/personal/sykorvo1/PPOthesis/ppo/gustySides/strength_[0.3, 0.4]/events.out.tfevents.1677864061.a11.2589115.0.v2",
        "/mnt/personal/sykorvo1/PPOthesis/ppo/gustySides/strength_[0.4, 0.5]/events.out.tfevents.1677959448.a10.1330885.0.v2",
        "/mnt/personal/sykorvo1/PPOthesis/ppo/BEST/noWind/events.out.tfevents.1670670309.a06.1146630.0.v2"
    ]
    
    more_lines_in_one_graph(input_paths, ax, colors, "Gusty wind form both sides", step_cutoff = 700)

def get_PRETgustySides(ax: plt.Axes, colors: List[str]) -> None:
    input_paths = [
        "/mnt/personal/sykorvo1/PPOthesis/ppo/PRETgustySides/strength_[0.3, 0.4]/events.out.tfevents.1677865502.a10.1276572.0.v2",
        "/mnt/personal/sykorvo1/PPOthesis/ppo/PRETgustySides/strength_[0.4, 0.5]/events.out.tfevents.1677959687.a11.2645933.0.v2",
        "/mnt/personal/sykorvo1/PPOthesis/ppo/gustySides/strength_[0.3, 0.4]/events.out.tfevents.1677864061.a11.2589115.0.v2",
        "/mnt/personal/sykorvo1/PPOthesis/ppo/gustySides/strength_[0.4, 0.5]/events.out.tfevents.1677959448.a10.1330885.0.v2",
        "/mnt/personal/sykorvo1/PPOthesis/ppo/BEST/noWind/events.out.tfevents.1670670309.a06.1146630.0.v2"
    ]
    PRETcolors =  ['#02b020', '#e8000b', '#7ce68e', '#db8186', '#8b2be2']
    PRETlabels = ['PRET strength_[0.3, 0.4]', 'PRET strength_[0.4, 0.5]', 'strength_[0.3, 0.4]', 'strength_[0.4, 0.5]', 'noWind']
    more_lines_in_one_graph(input_paths, ax, PRETcolors, "Gusty wind form both sides - Pretrained model without wind", PRETlabels, step_cutoff = 700)

# ----------------------------------------------------------------------

def make_multibar_plot(s1to2: List[int], s2to3: List[int], s3to4: List[int], s4to5: List[int], noWind: List[int], save_path: str = "multibar.png") -> None:
    """Makes a multibar plot with 5 different inputs

    Args:
        s1to2 : list of values for strength [0.1, 0.2] environment
        s2to3 : list of values for strength [0.2, 0.3] environment
        s3to4 : list of values for strength [0.3, 0.4] environment
        s4to5 : list of values for strength [0.4, 0.5] environment
        noWind : list of values for no wind environment
        save_path : path to save a figure of the graph. Defaults to "multibar.png".
    """
    barWidth = 0.15 # width of a bar
    fig, axs = plt.subplots()
    
    # Set position of bars on X axis
    br1 = np.arange(len(noWind))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]
    br5 = [x + barWidth for x in br4]
    
    # Plot the bars
    plt.bar(br1, s1to2,   color = colors[0], width = barWidth, edgecolor ='grey', label = 'strength_[0.1, 0.2]')
    plt.bar(br2, s2to3,   color = colors[1], width = barWidth, edgecolor ='grey', label = 'strength_[0.2, 0.3]')
    plt.bar(br3, s3to4,   color = colors[2], width = barWidth, edgecolor ='grey', label = 'strength_[0.3, 0.4]')
    plt.bar(br4, s4to5,   color = colors[3], width = barWidth, edgecolor ='grey', label = 'strength_[0.4, 0.5]')
    plt.bar(br5, noWind, color = colors[4], width = barWidth, edgecolor ='grey', label = 'noWind')
    
    # Add x axis labels
    plt.xlabel('Envs', fontweight ='bold', fontsize = 12)
    plt.ylabel('Average score over last 50 episodes', fontweight ='bold', fontsize = 12)
    plt.xticks([r + barWidth*2 for r in range(len(noWind))],
            ['noWind', 'left', 'gustyleft', 'right', 'gustyRight', 'sides', 'gustySides'])
    
    plt.legend(bbox_to_anchor=(0.85,0.6), bbox_transform=plt.gcf().transFigure)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.85)
    plt.savefig(save_path, bbox_inches="tight")

def make_example_multibar(save_path : str = "multibar.png") -> None:
    """Plots an example multibar and saves
    
    Args:    
        save_path: where to save the plot as a figure. Defaults to "multibar.png".
    """
    s1to2   = [2,1,3,2,1,2,3]
    s2to3   = [3,1,4,2,3,1,2]
    s3to4   = [4,2,1,3,1,2,4]
    s4to5   = [3,1,2,4,3,1,3] 
    noWind = [1,2,3,1,2,3,1]
    make_multibar_plot(s1to2, s2to3, s3to4, s4to5, noWind, save_path)


if __name__ == '__main__':
    
    # ------------------------
    # normal line/curve plots
    # ------------------------
    NORMAL_LINE = True
    if NORMAL_LINE:
        fig, axs = plt.subplots(ncols=1, nrows=1)
        
        # get_left(axs, colors)
        # get_right(axs, colors)
        # get_gustyLeft(axs, colors)
        # get_gustyRight(axs, colors)
        # get_sides(axs, colors)
        # get_gustySides(axs, colors)
        get_PRETgustySides(axs, colors)
        
        fig.supxlabel(f'steps ($10^{STEPS_POWER}$)')
        fig.supylabel('average score')
        fig.savefig("PRETgustySides.png")
    
    # -----------------------------------
    #   MultiBAR plots
    # -----------------------------------
    MULTIBAR = False
    if MULTIBAR:
        make_example_multibar()
        