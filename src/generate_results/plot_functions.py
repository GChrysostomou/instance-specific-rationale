import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import plotly.graph_objects as go
import plotly.offline as pyo


def export_legend(legend, filename="legend.png"):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)


def plot_best_var_VS_single_(datasets : list = ["sst", "multirc", "agnews", "evinf"],
    metrics_folder : str = "faithfulness_metrics",
    rationale_type: str = "topk",
    metric : str = "comprehensiveness",
    var_or_fixed : str = "fixed",
    plot_legend : bool = False,
    y_label_at_metric : str ="f1 macro avg - model labels",
    divergence : str = "jsd",
    save_to_dir : str = "graphs_and_tables"):


    
    mapper = {
        "gradients" : "x∇x",
        "ig" : "IG",
        "deeplift" : "DeepLift",
        "attention" : "α",
        "scaled attention" : "α∇α",
        "lime" : "LIME",
        "fixed-len_var-feat" : "OURS"
    }

    nicer_tasknames = {
        "sst" : "SST",
        "evinf" : "Ev.Inf.",
        "multirc" : "MultiRC",
        "agnews" : "AG"
    }

    assert metric in {"comprehensiveness", "sufficiency", "f1 macro avg - model labels", "f1 macro avg - actual labels"}, (
        """
        Must be one of the following metrics:
        * comprehensiveness
        * sufficiency
        * f1 macro avg - model labels
        * f1 macro avg - actual labels
        """
    )

    new_data_means = {}
    new_data_stds = {}

    for task_name in datasets:

        if "f1" in metric:

            fname = os.path.join(
                divergence, 
                metrics_folder,
                task_name,
                f"{rationale_type}-test-f1-metrics-description.json"
            )

        else:

            fname = os.path.join(
                divergence, 
                metrics_folder,
                task_name,
                f"{rationale_type}-test-faithfulness-metrics-description.json"
            )


        with open(fname, "r") as file : data = json.load(file) 

        new_data_means[task_name] = {}
        new_data_stds[task_name] = {}

        for feat_attr in ["deeplift", "lime", "attention", "scaled attention", "ig", "gradients", "fixed-len_var-feat"]:

            if "f1" in metric:

                new_data_means[task_name][mapper[feat_attr]] =  data[metric][f"{var_or_fixed}-{feat_attr}"]

            else:

                new_data_means[task_name][mapper[feat_attr]] =  data[f"{var_or_fixed}-{feat_attr}"][metric]["mean"]


    df = pd.DataFrame(new_data_means)

    df = df.rename(columns = nicer_tasknames)

    df = df.iloc[::-1]

    plt.style.use('tableau-colorblind10')
    # plt.style.use("seaborn-colorblind")
    plt.rcParams['axes.edgecolor']='#333F4B'
    plt.rcParams['axes.linewidth']=0.2
    plt.rcParams['xtick.color']='#333F4B'
    plt.rcParams['ytick.color']='#333F4B'
    plt.rcParams["font.variant"] = "small-caps"

    fig, ax = plt.subplots(figsize = (10,12))

    plt.yticks(fontsize = 35, rotation = 45)

    #     ax.set_xlabel(metric.capitalize(), fontsize = 35)

    df.T.plot.barh(ax = ax, legend=False, width=0.75)


    if metric != y_label_at_metric:

        y_axis = ax.axes.get_yaxis()
        y_axis.set_visible(False)
        ax.set_yticklabels([])


    bars = ax.patches
    hatches = ''.join(h*len(df.T) for h in 'x/O+*.')

    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    if "f1" not in metric:

        ax.set_xticks(np.arange(0.,df.max().max(), 0.2))
        plt.xticks(fontsize=23)
        ax.set_xticks(np.arange(0.,df.max().max()+0.025, 0.025), minor=True)


        if plot_legend:
            handles, labels = ax.get_legend_handles_labels()
            legend = plt.legend( 
                handles[::-1], labels[::-1],
                loc = "lower right",
                fontsize = 20, 
                fancybox = True, 
                framealpha = 0.4,
                labelspacing = 1,
                ncol=1,
                columnspacing=0.1,
                frameon=False
            )

    else:

        ax.set_xticks(np.arange(0,100+20,20))
        plt.xticks(fontsize=23)
        ax.set_xticks(np.arange(0,100,5), minor=True)

        if plot_legend:
            handles, labels = ax.get_legend_handles_labels()
            legend = plt.legend( 
                handles[::-1], labels[::-1],
                loc = (1,0), 
                fontsize = 20, 
                fancybox = True, 
                framealpha = 0.4,
                ncol=2,
                frameon=False
            )

    plt.grid(which='minor', alpha=0.15)
    plt.grid(which='major', alpha=0.5)
    plt.tight_layout()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


    plt.show()
    os.makedirs(f"{save_to_dir}/varying_feat_score/", exist_ok = True)

    fig.savefig(f"{save_to_dir}/varying_feat_score/{metric}-var-feat-score.png", dpi = 300)#, bbox_inches="tight")

    plt.close()
    
    print(f"*** Saved ours (best feat) vs every single feat --> {save_to_dir}/varying_feat_score/{metric}-var-feat-score.png")
    
    return


def plot_increasing_feat_(datasets : list = ["sst", "multirc", "agnews", "evinf"],
                            metrics_folder : str = "faithfulness_metrics",
                            rationale_type: str = "topk",
                            metric : str = "sufficiency",
                            var_or_fixed : str = "fixed",
                            plot_legend : bool = False,
                            y_label_at_metric : str = "f1 score - model labels"):


    assert metric in {"comprehensiveness", "sufficiency", "f1 score - model labels", "f1 score - actual labels"}, (
        """
        Must be one of the following metrics:
        * comprehensiveness
        * sufficiency
        * f1 score - model labels
        * f1 score - actual labels
        """
    )

    collective = {}

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

    mapper = {
        "gradients" : "x∇x",
        "ig" : "IG",
        "deeplift" : "DeepLift",
        "attention" : "α",
        "scaled attention" : "α∇α",
        "lime" : "LIME"
    }

    nicer_tasknames = {
        "sst" : "SST",
        "evinf" : "Ev.Inf.",
        "multirc" : "MultiRC",
        "agnews" : "AG"
    }



    df.index = ["\n".join([mapper[y] for y in x.split("-")]) for x in data[var_or_fixed].keys()]
    df = df.rename(columns = nicer_tasknames)
  
    df = df.iloc[::-1]

    plt.style.use('tableau-colorblind10')
    # plt.style.use("seaborn-colorblind")
    plt.rcParams['axes.edgecolor']='#333F4B'
    plt.rcParams['axes.linewidth']=0.2
    plt.rcParams['xtick.color']='#333F4B'
    plt.rcParams['ytick.color']='#333F4B'
    plt.rcParams["font.variant"] = "small-caps"

    fig, ax = plt.subplots(figsize = (8,10))

    plt.yticks(fontsize = 35, rotation = 45)

#     ax.set_xlabel(metric.capitalize(), fontsize = 35)
    
    df.T.plot.barh(ax = ax, legend=False, width=0.75)


    if metric != y_label_at_metric:

        y_axis = ax.axes.get_yaxis()
        y_axis.set_visible(False)
        ax.set_yticklabels([])


    bars = ax.patches
    hatches = ''.join(h*len(df.T) for h in 'x/O+*.')

    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    if "f1" not in metric:

        ax.set_xticks(np.arange(0.,df.max().max(), 0.2))
        plt.xticks(fontsize=23)
        ax.set_xticks(np.arange(0.,df.max().max()+0.025, 0.025), minor=True)


        if plot_legend:
            handles, labels = ax.get_legend_handles_labels()
            legend = plt.legend( 
                handles[::-1], labels[::-1],
                loc = (1,0), 
                fontsize = 20, 
                fancybox = True, 
                framealpha = 0.4,
                labelspacing = 1,
                ncol=2,
                columnspacing=0.1,
                frameon=False
            )

    else:

        ax.set_xticks(np.arange(0,100+20,20))
        plt.xticks(fontsize=23)
        ax.set_xticks(np.arange(0,100,5), minor=True)

        if plot_legend:
            handles, labels = ax.get_legend_handles_labels()
            legend = plt.legend( 
                handles[::-1], labels[::-1],
                loc = (1,0), 
                fontsize = 20, 
                fancybox = True, 
                framealpha = 0.4,
                ncol=2,
                frameon=False
            )
    
    plt.grid(which='minor', alpha=0.15)
    plt.grid(which='major', alpha=0.5)
    plt.tight_layout()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    if plot_legend:
        
        export_legend(legend, filename = "graphs_and_tables/increasing-set-of-feature_scoring/legend.png")
    
    plt.show()
    os.makedirs("graphs_and_tables/increasing-set-of-feature_scoring/", exist_ok = True)
    
    fig.savefig(f"graphs_and_tables/increasing-set-of-feature_scoring/{metric}-increasing-feat-attr.png", dpi = 300)#, bbox_inches="tight")

    plt.close()

    return

def plot_at_different_N_(
    save_to_dir : str = "graphs_and_tables",
    single_N : str = "jsd/faithfulness_metrics",
    double_N : str = "double_jsd/faithfulness_metrics",
    rationale_type : str = "topk"
):


    keep = {
        "fixed-deeplift" : {
            "marker" : "v",
            "name" : "DeepLift"
        },
        "fixed-ig" : {
            "marker" : ">",
            "name" : "IG"
        },
        "fixed-lime" : {
            "marker" : "^",
            "name" : "LIME"
        },
        "fixed-attention" : {
            "marker" : "<",
            "name" : "α"
        },
        "fixed-scaled attention" : {
            "marker" : "d",
            "name" : "α∇α"
        },
        "fixed-gradients": {
            "marker" : "D",
            "name" : "x∇x"
        },
        "var-var-len_var-feat_var-type" : {
            "marker" : "X",
            "name" : "OURS"
        }
    }

    nicer_tasknames = {
        "sst" : "SST",
        "evinf" : "Ev.Inf.",
        "multirc" : "MultiRC",
        "agnews" : "AG"
    }

    plt.style.use('tableau-colorblind10')

    plt.rcParams['axes.edgecolor']='#333F4B'
    plt.rcParams['axes.linewidth']=0.2
    plt.rcParams['xtick.color']='#333F4B'
    plt.rcParams['ytick.color']='#333F4B'
    plt.rcParams["font.variant"] = "small-caps"

    fig, ax = plt.subplots(2,1, figsize = (8,8))

    for _i_, metric in enumerate(["comprehensiveness", "sufficiency"]):

        for task_name in ["sst", "evinf", "agnews", "multirc"]:


            ratios = [0.2, 0.4]

            fname_short = f"{single_N}/{task_name}/{rationale_type}-test-faithfulness-metrics-description.json"

            with open(fname_short, "r") as file : data_short = json.load(file) 

            fname_long = f"{double_N}/{task_name}/{rationale_type}-test-faithfulness-metrics-description.json"

            with open(fname_long, "r") as file : data_long = json.load(file) 

            ax = ax.flatten()


            for feat_attr in ["var-var-len_var-feat_var-type"]:

                if feat_attr in keep.keys():

                    short = data_short[feat_attr][metric]
                    long = data_long[feat_attr][metric]

                    ax[_i_].plot(
                        ratios,
                        [short["mean"], long["mean"]],
                        marker=keep[feat_attr]["marker"],
                        markersize = 15,
                        linewidth = 3,
                        label=nicer_tasknames[task_name]
                    )

                    ax[_i_].fill_between(
                        ratios,
                        [short["mean"]-short["std"]*0.05, long["mean"]-long["std"]*0.05],
                        [short["mean"]+short["std"]*0.05, long["mean"]+long["std"]*0.05],
                        alpha=0.2
                    )    

        if _i_ == 1:

            plt.xticks(ratios)
            ax[_i_].set_xticklabels(["1x", "2x"], fontsize=15)

            plt.yticks(fontsize=15)


            plt.tight_layout()

        else:

            ax[_i_].legend(loc = "lower right", fontsize = 20)
            ax[_i_].set_xticks([], [])
            ax[_i_].set_yticklabels([round(x,1) for x in np.arange(0.0, 0.9, 0.1)], fontsize=15)


        ax[_i_].set_ylabel(metric.capitalize(), fontsize = 25)
    
    os.makedirs(f"{save_to_dir}/doubling_the_length/", exist_ok = True)

    fig.savefig(f"{save_to_dir}/doubling_the_length/all_tasks.png", dpi = 300, bbox_inches="tight")
    
    print(f"** figures saved in --> {save_to_dir}/doubling_the_length/")
    
    return

def plot_radars_(datasets : list = ["sst", "multirc", "agnews", "evinf"],
    metrics_folder : str = "faithfulness_metrics",
    rationale_type: str = "topk",
    metric : str = "f1 macro avg - model labels",
    var_or_fixed : str = "fixed",
    plot_legend : bool = False,
    y_label_at_metric : str ="f1 macro avg - model labels",
    divergence : str = "jsd"):



    mapper = {
        "gradients" : "x∇x",
        "ig" : "IG",
        "deeplift" : "DeepLift",
        "attention" : "α",
        "scaled attention" : "α∇α",
        "lime" : "LIME",
        "fixed-len_var-feat" : "OURS"
    }

    nicer_tasknames = {
        "sst" : "SST",
        "evinf" : "Ev.Inf.",
        "multirc" : "MultiRC",
        "agnews" : "AG"
    }

    assert metric in {"comprehensiveness", "sufficiency", "f1 macro avg - model labels", "f1 macro avg - actual labels"}, (
        """
        Must be one of the following metrics:
        * comprehensiveness
        * sufficiency
        * f1 macro avg - model labels
        * f1 macro avg - actual labels
        """
    )


    collect_dfs = {}

    for metric in {"comprehensiveness", "sufficiency", "f1 macro avg - model labels"}:

        new_data_means = {}

        for task_name in datasets:

            if "f1" in metric:

                fname = os.path.join(
                    divergence,
                    metrics_folder,
                    task_name,
                    f"{rationale_type}-test-f1-metrics-description.json"
                )

            else:

                fname = os.path.join(
                    divergence,
                    metrics_folder,
                    task_name,
                    f"{rationale_type}-test-faithfulness-metrics-description.json"
                )


            with open(fname, "r") as file : data = json.load(file) 

            new_data_means[task_name] = {}

            for feat_attr in ["deeplift", "lime", "attention", "scaled attention", "ig", "gradients", "fixed-len_var-feat"]:

                if "f1" in metric:

                    new_data_means[task_name][mapper[feat_attr]] =  data[metric][f"{var_or_fixed}-{feat_attr}"]

                else:

                    new_data_means[task_name][mapper[feat_attr]] =  data[f"{var_or_fixed}-{feat_attr}"][metric]["mean"]


        df = pd.DataFrame(new_data_means)

        df = df.rename(columns = nicer_tasknames)

        df = df.iloc[::-1]

        df["metric"] = metric

        collect_dfs[metric] = df

    df = pd.concat(collect_dfs.values())
    df["feat_attr"] = df.index

    layout = go.Layout(
      margin=go.layout.Margin(
            l=0, #left margin
            r=0, #right margin
            b=0, #bottom margin
            t=0, #top margin
        )
    )

    fig = dict(data=data, layout=layout)

    for metric in {"sufficiency", "comprehensiveness", "f1 macro avg - model labels"}:

        legend_show = False#True if metric == "comprehensiveness" else False 

        colors = {
            "x∇x" : "red",
            "IG" : "orange",
            "DeepLift" : "orangered",
            "α" : "green",
            "α∇α" : "olive",
            "LIME" : "darkgrey"

        }

        marker_symbol = {
            "x∇x" : "triangle-down",
            "IG" : "triangle-left",
            "DeepLift" : "triangle-right",
            "α" : "square",
            "α∇α" : "star-square",
            "LIME" : "hourglass"

        }

        categories = [nicer_tasknames[x] for x in datasets]
        categories = [*categories, categories[0]]

        sor_data = []

        for feat_attr in mapper.values():


            temp = df[(df.feat_attr == feat_attr) & (df.metric == metric)][[nicer_tasknames[x] for x in datasets]].values[0]

            temp = [*temp, temp[0]]

            if feat_attr == "OURS":

                sor_data.append(go.Scatterpolar(r=np.asarray(temp), theta=categories, name=feat_attr,line = {
                    "color" : "blue"}, marker_symbol="circle-cross", marker_size=15))

            else:
                sor_data.append(go.Scatterpolar(r=np.asarray(temp), theta=categories, name=feat_attr, line = {
                    "color" : colors[feat_attr], "dash": "dash"}, marker_symbol=marker_symbol[feat_attr],
                                               marker_size=15))

        fig = go.Figure(
            data=sor_data,
            layout=go.Layout(
                polar={'radialaxis': {'visible': True}},
                showlegend=legend_show,
                font = {"size": 26},
                margin=go.layout.Margin(
                    l=20, #left margin
                    r=35, #right margin
                    b=35, #bottom margin
                    t=35, #top margin
                )
            )
        )

        plt.tight_layout()

        os.makedirs(f"graphs_and_tables/varying_feat_score_RADAR/{rationale_type}", exist_ok = True)

        if legend_show:

            fig.write_image(
                f"graphs_and_tables/varying_feat_score_RADAR/{rationale_type}/legend-var-feat-score.png", 
                width=670, 
                height=610, 
                scale=5
            )

        else:

            fig.write_image(
                f"graphs_and_tables/varying_feat_score_RADAR/{rationale_type}/{metric}-var-feat-score.png", 
                width=650, 
                height=620, 
                scale=5
            )

    print(f"*** radar plots saved in --> graphs_and_tables/varying_feat_score_RADAR/{rationale_type}")

    return