import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import plotly.graph_objects as go


def export_legend(legend, filename="legend.png"):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)


def plot_increasing_feat_(
    datasets : list = ["sst", "multirc", "agnews", "evinf"],
    metrics_folder : str = "jsd/faithfulness_metrics",
    rationale_type: str = "topk",
    metric : str = "sufficiency",
    var_or_fixed : str = "fixed",
    plot_legend : bool = False,
    y_label_at_metric : str = "f1 score - model labels",
    you_want_all : bool = False,
    add_even_the_best : bool = False):


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


    keep_only_this = {
        'gradients' : 'gradients',
        'gradients-ig' : 'gradients-ig',
        'gradients-deeplift-ig' : 'gradients-ig-deeplift',
        'gradients-deeplift-attention-ig' : 'gradients-ig-deeplift-attention',
        'gradients-deeplift-scaled attention-attention-ig': 'gradients-ig-deeplift-attention-scaled attention',
        'gradients-deeplift-scaled attention-lime-attention-ig' : 'gradients-ig-deeplift-attention-scaled attention-lime'
    }

    df.index = [x for x in data[var_or_fixed].keys()]

    if you_want_all  == False:

        df = df[df.index.isin(keep_only_this)]
        
        df["temp"] = df.index

        df["temp"] = df.temp.apply(lambda x : keep_only_this[x])

        df.index = df.temp

        df = df.drop(columns = ["temp"])

    df.index = ["{"+"\n".join([mapper[y] for y in x.split("-")])+"}" for x in df.index]

    df = df.rename(columns = nicer_tasknames)

    if add_even_the_best:
    
        extra = pd.read_csv(f"graphs_and_tables/var_all_table/{rationale_type}-all.csv")
        
        if "f1" in metric:

            extra = extra[[x for x in extra.columns if "f1" in x] + ["Unnamed: 0"]]

        else:

            extra = extra[[x for x in extra.columns if metric in x] + ["Unnamed: 0"]]
        
        extra = extra[extra["Unnamed: 0"] == "fixed"].drop(columns = ["Unnamed: 0"]).values[0]

        
        df = pd.concat([df, pd.DataFrame(dict(zip(df.columns, extra)),  index =["{Best F.S.}"])], axis = 0)

    plt.style.use('tableau-colorblind10')
    plt.rcParams['axes.edgecolor']='#333F4B'
    plt.rcParams['axes.linewidth']=0.2
    plt.rcParams['xtick.color']='#333F4B'
    plt.rcParams['ytick.color']='#333F4B'
    plt.rcParams["font.variant"] = "small-caps"

    fig, ax = plt.subplots(figsize = (8,10))

    plt.yticks(fontsize = 35, rotation = 45)
    
    
    df.T.plot.barh(ax = ax, legend=False, width=0.88)

    
    if metric != y_label_at_metric:

        y_axis = ax.axes.get_yaxis()
        y_axis.set_visible(False)
        ax.set_yticklabels([])


    bars = ax.patches

    hatches = ['/', 'O', '*', '+', 'x', 'o', '*', '.', '|','/', 'O', '-', '+', 'x', 'o', "*"][:len(df)]

    hatches = ''.join(h*4 for h in hatches)

    leng = len(hatches)
    for j_, (bar, hatch) in enumerate(zip(bars, hatches)):

        if add_even_the_best:
            
            if j_ > (leng-5):

                bar.set_color("lightblue")

        bar.set_hatch(hatch)
    
    if you_want_all:
    
        kol = 3
        
    else:
        
        kol = 2
    
    if "f1" not in metric:

        ax.set_xticks(np.arange(0.,df.max().max(), 0.2))
        plt.xticks(fontsize=23)
        ax.set_xticks(np.arange(0.,df.max().max()+0.025, 0.025), minor=True)


        if plot_legend:
            handles, labels = ax.get_legend_handles_labels()
            legend = plt.legend( 
                handles[::-1], labels[::-1],
                loc = (1,0), 
                fontsize = 30, 
                fancybox = True, 
                framealpha = 0.4,
                labelspacing = 1.2,
                ncol=kol,
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
                ncol=kol,
                frameon=False
            )

    plt.grid(which='minor', alpha=0.15)
    plt.grid(which='major', alpha=0.5)
    plt.tight_layout()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if plot_legend:
        
        if you_want_all:
            
            export_legend(legend, filename = "graphs_and_tables/increasing-set-of-feature_scoring/legend-EXTRA.png")
        
        else:
            
            export_legend(legend, filename = "graphs_and_tables/increasing-set-of-feature_scoring/legend.png")
        
        return 
    
    plt.show()
    
    os.makedirs("graphs_and_tables/increasing-set-of-feature_scoring/", exist_ok = True)
    
    if you_want_all:
        
        fig.savefig(f"graphs_and_tables/increasing-set-of-feature_scoring/{metric}-EXTRA-increasing-feat-attr.png", dpi = 300)#, bbox_inches="tight")
    else:
        
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

    for _i_, metric in enumerate(["comprehensiveness", "sufficiency", "f1 macro avg - model labels"]):
        
        
        fig, ax = plt.subplots(1,1, figsize = (5,7))

        for task_name in ["sst", "evinf", "agnews", "multirc"]:


            ratios = [0.2, 0.4]
            
            if "f1" in metric:
                
                fname_short = f"{single_N}/{task_name}/{rationale_type}-test-f1-metrics-description.json"

                with open(fname_short, "r") as file : data_short = json.load(file) 

                fname_long = f"{double_N}/{task_name}/{rationale_type}-test-f1-metrics-description.json"

                with open(fname_long, "r") as file : data_long = json.load(file) 
                
            else:

                fname_short = f"{single_N}/{task_name}/{rationale_type}-test-faithfulness-metrics-description.json"

                with open(fname_short, "r") as file : data_short = json.load(file) 

                fname_long = f"{double_N}/{task_name}/{rationale_type}-test-faithfulness-metrics-description.json"

                with open(fname_long, "r") as file : data_long = json.load(file) 

            for feat_attr in ["var-var-len_var-feat_var-type"]:

                if feat_attr in keep.keys():
                    
                    if "f1" in metric:
                        
                        short = data_short["f1 macro avg - model labels"][feat_attr]
                        long = data_long["f1 macro avg - model labels"][feat_attr]

                        ax.plot(
                            ratios,
                            [short, long],
                            marker=keep[feat_attr]["marker"],
                            markersize = 15,
                            linewidth = 3,
                            label=nicer_tasknames[task_name]
                        )
                    
                    
                    else:
                        
                        short = data_short[feat_attr][metric]
                        long = data_long[feat_attr][metric]

                        ax.plot(
                            ratios,
                            [short["mean"], long["mean"]],
                            marker=keep[feat_attr]["marker"],
                            markersize = 15,
                            linewidth = 3,
                            label=nicer_tasknames[task_name]
                        )

                        ax.fill_between(
                            ratios,
                            [short["mean"]-short["std"]*0.05, long["mean"]-long["std"]*0.05],
                            [short["mean"]+short["std"]*0.05, long["mean"]+long["std"]*0.05],
                            alpha=0.2
             
                        )
        
        
        plt.xticks(ratios)
        ax.set_xticklabels(["1x", "2x"], fontsize=25)

        plt.yticks(fontsize=20)


        plt.tight_layout()


        if "f1" in metric:

            ax.legend(
                loc = "upper right",
                fontsize = 20
            )

        else:

#             ax.set_xticks([], [])
            ax.set_yticklabels([round(x,1) for x in np.arange(0.0, 0.9, 0.1)], fontsize=20)

        
        if metric == "sufficiency":
            
            metr = "NormSuff"
            
        elif "f1" in metric:
            
            metr = "F1"
            
        else:
            
            metr = "NormComp"

            
#         ax.set_ylabel(metr, fontsize = 25)
    
        os.makedirs(f"{save_to_dir}/doubling_the_length/", exist_ok = True)

        fig.savefig(f"{save_to_dir}/doubling_the_length/{metric}.png", dpi = 300, bbox_inches="tight")
        
        plt.close()

    print(f"** figures saved in --> {save_to_dir}/doubling_the_length/")
    
    return

def plot_radars_(
    datasets : list = ["sst", "multirc", "agnews", "evinf"],
    metrics_folder : str = "faithfulness_metrics",
    rationale_type: str = "topk",
    metric_list : str = ["f1 macro avg - model labels", "sufficiency", "comprehensiveness"],
    var_or_fixed : str = "fixed",
    divergence : str = "jsd",
    legend_show : bool = False):



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

  
    collect_dfs = {}

    for metric in metric_list:

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

    for metric in metric_list:

        colors = {
            "x∇x" : "lime",
            "IG" : "orange",
            "DeepLift" : "purple",
            "α" : "green",
            "α∇α" : "orangered",
            "LIME" : "darkgrey",
            "Random" : "black"

        }
        
        dashed = {
            "x∇x" : "dot",
            "IG" : "dot",
            "DeepLift" : "longdash",
            "α" : "longdashdot",
            "α∇α" : "longdashdot",
            "LIME" : "longdash",
            "Random" : "dashdot"    
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
                    "color" : colors[feat_attr], "dash": dashed[feat_attr]}, marker_symbol=marker_symbol[feat_attr],
                                               marker_size=15))

        fig = go.Figure(
            data=sor_data,
            layout=go.Layout(
                polar={'radialaxis': {'visible': True}, 'bgcolor' : 'aliceblue'},
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

    if legend_show:

        print(f"*** LEGEND FOR -> radar plots saved in --> graphs_and_tables/varying_feat_score_RADAR/{rationale_type}")

    else:

        print(f"*** radar plots saved in --> graphs_and_tables/varying_feat_score_RADAR/{rationale_type}")

    return
