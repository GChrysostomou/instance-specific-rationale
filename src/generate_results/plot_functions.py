import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

<<<<<<< HEAD
<<<<<<< HEAD
def plot_increasing_feat_(datasets : set = {"sst", "evinf", "multirc"}, 
=======
=======
>>>>>>> 0b8d801278e68125ff649ef4e5ac17371c71fb35
plt.style.use('tableau-colorblind10')
plt.rcParams['axes.edgecolor']='#333F4B'
plt.rcParams['axes.linewidth']=0.2
plt.rcParams['xtick.color']='#333F4B'
plt.rcParams['ytick.color']='#333F4B'
plt.rcParams["font.variant"] = "small-caps"

def plot_increasing_feat_attr_(datasets : set = {"sst", "evinf", "multirc"}, 
<<<<<<< HEAD
>>>>>>> added reqs
=======
>>>>>>> 0b8d801278e68125ff649ef4e5ac17371c71fb35
                          metrics_folder : str = "faithfulness_metrics",
                          rationale_type: str = "topk",
                          metric : str = "comprehensiveness",
                          var_or_fixed : str = "fixed"):

    collective = {}

    SAVE_DIR=os.path.join(
        os.getcwd(),
        "graphs_and_tables",
        "plots",
        ""
    )

    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"***** plots saved in -> {SAVE_DIR}")

    for task_name in datasets:
        
        fname = os.path.join(
            metrics_folder,
            task_name,
            f"{rationale_type}-test-increasing-feature-scoring.json"
        )


        with open(fname, "r") as file : data = json.load(file) 
        
        if "f1" not in metric:

            collective[task_name] = [x[metric]["mean"] for x in data[var_or_fixed].values()]

        else:

            collective[task_name] = [x[metric] for x in data[var_or_fixed].values()]


    df = pd.DataFrame(collective)

    df.index = ["\n".join(x.split("-")) for x in data[var_or_fixed].keys()]
    df = df.iloc[::-1]

<<<<<<< HEAD
<<<<<<< HEAD


    plt.style.use('tableau-colorblind10')
    # plt.style.use("seaborn-colorblind")
    plt.rcParams['axes.edgecolor']='#333F4B'
    plt.rcParams['axes.linewidth']=0.2
    plt.rcParams['xtick.color']='#333F4B'
    plt.rcParams['ytick.color']='#333F4B'
    plt.rcParams["font.variant"] = "small-caps"

=======
    ## plotting
>>>>>>> added reqs
=======
    ## plotting
>>>>>>> 0b8d801278e68125ff649ef4e5ac17371c71fb35
    fig, ax = plt.subplots(figsize = (14,10))

    plt.yticks(fontsize = 15)

    ax.set_xlabel(metric, fontsize = 25)
    ax.set_ylabel("Feat-Attr Combo", fontsize = 25)


    df.plot.barh(ax = ax)

    if "f1" not in metric:

        plt.xticks(ticks = [0, 0.2,0.4,0.6, 0.8] ,fontsize = 23)

        ax.legend( 
            fontsize = 20, 
            fancybox = True, 
            framealpha = 0.4
        )

    else:

        plt.xticks(ticks = [0, 20,40,60,80,100] ,fontsize = 23)


        ax.legend(
            loc = (0.9,0), 
            fontsize = 20, 
            fancybox = True, 
            framealpha = 0.4
        )

    plt.grid()
    plt.tight_layout()

    fig.savefig(f"{SAVE_DIR}{metric}-increasing-feat-attr.png", dpi = 300, bbox_inches="tight")

    plt.close()
    
    return