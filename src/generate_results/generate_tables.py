import json
import pandas as pd
import numpy as np
import os
from scipy.stats import wilcoxon

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


def create_table_of_rationale_lengths_(divergence: str, extracted_rationale_dir : str = "extracted_rationales", 
                                     save_to_dir : str = "graphs_and_tables/", double : bool = False):

    ratio_data = {}
    ratio_means = {}

    for thresh in ["topk", "contigious"]:

        for task_name in ["sst", "agnews", "evinf", "multirc"]:

            nicer_task_name = nicer_tasknames[task_name]

            if double:
                
                fname = f"double_{divergence}/{extracted_rationale_dir}/{task_name}/{thresh}/test-rationale_metadata.npy"

            else:

                fname = f"{divergence}/{extracted_rationale_dir}/{task_name}/{thresh}/test-rationale_metadata.npy"

            data = np.load(fname, allow_pickle = True).item()

            ratio_data[f"{nicer_task_name}-{thresh}"] = {}
            ratio_means[f"{nicer_task_name}-{thresh}"] = {}

            for feat_attr in ["gradients", "ig", "deeplift", "lime", "attention", "scaled attention"]:

                get_ratios = []

                for key in data:

                    get_ratios.append(data[key][feat_attr]["variable rationale ratio"])


                feat_attr = mapper[feat_attr]

                get_ratios = np.asarray(get_ratios)

                ratio_data[f"{nicer_task_name}-{thresh}"][feat_attr] = f"{round(get_ratios.mean() * 100,1)} ± {round(get_ratios.std() * 100,0)}"
                ratio_means[f"{nicer_task_name}-{thresh}"][feat_attr] = get_ratios.mean() * 100


    df = pd.DataFrame(ratio_data).T

    means = pd.DataFrame(ratio_means).T.mean(1).to_dict()

    df["avg."] = pd.DataFrame(ratio_means).T.mean(1).round(1)
    
    folder_name = os.path.join(
        save_to_dir, 
        "rationale_lengths",
        ""
    )
    
    os.makedirs(
        folder_name,
        exist_ok=True
    )

    if double:
    
        df.to_latex(f"{folder_name}{divergence}-rationale_lengths-2N.tex")

    else:

        df.to_latex(f"{folder_name}{divergence}-rationale_lengths.tex")

    print(f"** Rationale lengths saved in --> {folder_name}{divergence}-rationale_lengths.tex")
    
    return

def generate_table_for_var_combos_(
    datasets : list = ["sst", "multirc", "agnews", "evinf"],
    metrics : list = ["f1 macro avg - model labels", "sufficiency", "comprehensiveness"],
    metrics_folder : str = "faithfulness_metrics",
    rationale_type: str = "contigious",
    divergence : str = "jsd",
    save_to_dir : str = "graphs_and_tables"):

    mapper = {
        "fixed-fixed-len_var-feat" : "Fixed-Length + Var.-Feat",
        "fixed-fixed-len_var-feat_var-type" : "Fixed-Length + Var.-Feat + Var.-Type",
        "var-var-len_var-feat" : "Var.-Length + Var.-Feat",
        "var-var-len_var-feat_var-type" : "Var.-Length + Var.-Feat + Var.-Type",

    }

    nicer_tasknames = {
        "sst" : "SST",
        "evinf" : "Ev.Inf.",
        "multirc" : "MultiRC",
        "agnews" : "AG"
    }

    new_data_means = {}

    maxes = {}

    for metric in metrics:

        for task_name in datasets:

            for rationale_type in [rationale_type]:#, "contigious"]:


                task_name_nicer = nicer_tasknames[task_name]


                maxes[f"{task_name_nicer}-{metric}-{rationale_type}"] = {}
                new_data_means[f"{task_name_nicer}-{metric}-{rationale_type}"] = {}

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

                for feat_attr in ["fixed-fixed-len_var-feat", 
                                  "fixed-fixed-len_var-feat_var-type", 
                                  "var-var-len_var-feat", 
                                  "var-var-len_var-feat_var-type"]:

                    if "f1" in metric:

                        new_data_means[f"{task_name_nicer}-{metric}-{rationale_type}"][f"{mapper[feat_attr]}"] =  round(data[metric][f"{feat_attr}"],1)

                    else:

                        new_data_means[f"{task_name_nicer}-{metric}-{rationale_type}"][f"{mapper[feat_attr]}"] =  round(data[f"{feat_attr}"][metric]["mean"],3)


                for var_or_fixed in ["fixed", "var"]:

                    temp = {}

                    for feat_attr in ["deeplift", "lime", "attention", "scaled attention", "ig", "gradients"]:

                        if "f1" in metric:

                            temp[feat_attr] =  data[metric][f"{var_or_fixed}-{feat_attr}"]

                            fun = min

                        else:

                            temp[feat_attr] =  data[f"{var_or_fixed}-{feat_attr}"][metric]["mean"]

                            fun = max


                    maxes[f"{task_name_nicer}-{metric}-{rationale_type}"][var_or_fixed] = fun(temp.values())

    detailed = pd.DataFrame(new_data_means)#.round(3)

    os.makedirs(f"{save_to_dir}/var_all_table/", exist_ok = True)

    df = pd.concat([pd.DataFrame(maxes), detailed]).round(2)

    df.to_latex(f"{save_to_dir}/var_all_table/{rationale_type}-all.tex", escape=False)
    
    print(f"*** Var combos saved in -> {save_to_dir}/var_all_table/{rationale_type}*")
    
    return df

def generate_table_for_divergence_(save_to_dir : str = "graphs_and_tables"):

    new_data = {}

    for thresh in ["topk", "contigious"]:

        for divergence in ["jsd", "kldiv", "classdiff", "perplexity"]:

            df = generate_table_for_var_combos_(
                save_to_dir=None, 
                metrics = ["sufficiency", "comprehensiveness", "f1 macro avg - model labels"],
                divergence = divergence,
                rationale_type = thresh
            )

            df = df[df.index == "Var.-Length + Var.-Feat + Var.-Type"]

            new_data[f"{divergence}-{thresh}"] = {}

            for metric in ["sufficiency", "comprehensiveness", "f1 macro avg - model labels"]:

                filtered = df[[x for x in df.columns if metric in x]]

                new_data[f"{divergence}-{thresh}"][metric] = filtered.mean(1).item()

    df = pd.DataFrame(new_data).T.round(2)
    
    os.makedirs(f"{save_to_dir}/divergence_all/", exist_ok = True)
    
    df.to_latex(f"{save_to_dir}/divergence_all/divergence-var-feat-var-len.tex", escape=False)
    
    print(f"*** Divergence comparison saved in -> {save_to_dir}/divergence_all/")
    
    return

def generate_time_tables_(
    full_fidel_folder : str = "full_fidelity", 
    two_perc_foler: str = "jsd", 
    five_perc_folder : str = "five_percent", 
    save_to_dir : str = "graphs_and_tables/"):

    time_data = {}
    faith_data = {}

    for fidelity in ["@each token", "@2%", "@5%"]:

        time_data[f"{fidelity} (s)"] = {}
        faith_data[f"F1 macro - {fidelity}"] = {}
        faith_data[f"Sufficiency - {fidelity}"] = {}
        faith_data[f"Comprehensiveness - {fidelity}"] = {}
        faith_data[f"Computed Length - {fidelity}"] = {}

        for thresh in ["topk", "contigious"]:

            for task_name in ["sst", "agnews", "evinf", "multirc"]:

                if fidelity == "@each token":

                    fname = f"{full_fidel_folder}/extracted_rationales/{task_name}/{thresh}/test-rationale_metadata.npy"
                    f1 = f"{full_fidel_folder}/faithfulness_metrics/{task_name}/{thresh}-test-f1-metrics-description.json"
                    suff_comp = f"{full_fidel_folder}/faithfulness_metrics/{task_name}/{thresh}-test-faithfulness-metrics-description.json"

                elif fidelity == "@2%":

                    fname = f"{two_perc_foler}/extracted_rationales/{task_name}/{thresh}/test-rationale_metadata.npy"
                    f1 = f"{two_perc_foler}//faithfulness_metrics/{task_name}/{thresh}-test-f1-metrics-description.json"
                    suff_comp = f"{two_perc_foler}//faithfulness_metrics/{task_name}/{thresh}-test-faithfulness-metrics-description.json"


                else:

                    fname = f"{five_perc_folder}/extracted_rationales/{task_name}/{thresh}/test-rationale_metadata.npy"
                    f1 = f"{five_perc_folder}/faithfulness_metrics/{task_name}/{thresh}-test-f1-metrics-description.json"
                    suff_comp = f"{five_perc_folder}/faithfulness_metrics/{task_name}/{thresh}-test-faithfulness-metrics-description.json"

                data = np.load(fname, allow_pickle = True).item()

                with open(f1, "r") as file : f1 = json.load(file)
                with open(suff_comp, "r") as file : suff_comp = json.load(file)

                get_time = []
                get_ratio = []
                
                task_name = nicer_tasknames[task_name]

                for key in data:

                    get_time.append(data[key]["attention"]["time elapsed"])
                    get_ratio.append(data[key]["var-len_var-feat"]["variable rationale ratio"])


                get_time = np.asarray(get_time)
                get_ratio = np.asarray(get_ratio)


                time_data[f"{fidelity} (s)"][f"{task_name}-{thresh}"] = get_time.mean()

                faith_data[f"F1 macro - {fidelity}"][f"{task_name}-{thresh}"] = f1['f1 macro avg - model labels']["var-var-len_var-feat"]
                faith_data[f"Sufficiency - {fidelity}"][f"{task_name}-{thresh}"] = suff_comp["var-var-len_var-feat"]["sufficiency"]["mean"]
                faith_data[f"Comprehensiveness - {fidelity}"][f"{task_name}-{thresh}"] = suff_comp["var-var-len_var-feat"]["comprehensiveness"]["mean"]
                faith_data[f"Computed Length - {fidelity}"][f"{task_name}-{thresh}"]= get_ratio.mean() * 100

    df = pd.DataFrame(time_data).round(2)

    df["R.I. @ 2%"] = (df["@each token (s)"] / df["@2% (s)"]).round(1)
    df["R.I. @ 5%"] = (df["@each token (s)"] / df["@5% (s)"]).round(1)

    df2 = pd.DataFrame(faith_data).round(2)

    folder_name = os.path.join(
        save_to_dir, 
        "time_for_var_length",
        ""
    )
    
    os.makedirs(
        folder_name,
        exist_ok=True
    )
    
    df = df[["@each token (s)", "@2% (s)", "R.I. @ 2%", "@5% (s)", "R.I. @ 5%"]]

    df.to_latex(f"{folder_name}time-taken-for-var-rationales.tex")
    df2.to_latex(f"{folder_name}faith-across-skips.tex")

    print(f"** Rationale lengths saved in --> {folder_name}time-taken-for-var-rationales.tex")
    
    return 

def significance_results_(datasets : list = ["sst", "multirc", "agnews", "evinf"],
    metrics_folder : str = "faithfulness_metrics",
    rationale_type: str = "topk",
    metric : str = "f1 macro avg - model labels",
    var_or_fixed : str = "fixed",
    save_to_dir : str = "graphs_and_tables",
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


    compare = {}

    for metric in {"sufficiency"}:

        new_data_means = {}

        for task_name in datasets:

            compare[task_name] = {}

            fname = os.path.join(
                divergence,
                metrics_folder,
                task_name,
                f"{rationale_type}-test-faithfulness-metrics.json"
            )


            with open(fname, "r") as file : data = json.load(file) 

            new_data_means[task_name] = {}

            for feat_attr in ["deeplift", "lime", "attention", "scaled attention", "ig", "gradients", "fixed-len_var-feat"]:

                new_data_means[task_name][feat_attr] = []

                for annotation_id in data:

                    new_data_means[task_name][feat_attr].append(
                        data[annotation_id][f"{var_or_fixed}-{feat_attr}"][metric]
                    )


            for feat_attr in ["deeplift", "lime", "attention", "scaled attention", "ig", "gradients"]:

                pval = wilcoxon(
                    new_data_means[task_name]["fixed-len_var-feat"],
                    new_data_means[task_name][feat_attr]
                ).pvalue

                mean_ours = np.asarray(new_data_means[task_name]["fixed-len_var-feat"]).mean()
                mean_fixed = np.asarray(new_data_means[task_name][feat_attr]).mean()

                if  mean_ours >  mean_fixed:

                    if pval < 0.05:

                        string = "HIGHER - SIGNIFICANT"

                    else:

                        string = "HIGHER - NOT SIGNIFICANT"

                else:

                    if pval < 0.05:

                        string = "LOWER - SIGNIFICANT"

                    else:

                        string = "LOWER - NOT SIGNIFICANT"

                compare[task_name][f"OURS-Vs-{feat_attr}"] = string

    df = pd.DataFrame(compare)
    
    folder_name = os.path.join(
        save_to_dir, 
        "significance_var_feat",
        ""
    )
    
    os.makedirs(
        folder_name,
        exist_ok=True
    )
   
    df.to_latex(f"{folder_name}-sig.tex")
    
    print(f"** Significance tests saved in --> {folder_name}-sig.tex")
                
    return


### WITH ARROWS
def make_tables_for_rationale_length_var_ARROWS_(datasets : list = ["sst", "multirc", "agnews", "evinf"],
    metrics_folder : str = "faithfulness_metrics",
    rationale_type: str = "contigious",
    save_to_dir : str = "graphs_and_tables",
    divergence : str= "jsd"):

    raise NotImplementedError(
        """
        Not used anymore, uncomment this error if you want to check results with arrows
        """
    )

    mapper = {
        "gradients" : "$\boldsymbol\mathbf{x}\nabla\mathbf{x}$",
        "ig" : "\textbf{IG}",
        "deeplift" : "\textbf{DeepLift}",
        "attention" : "$\boldsymbol\alpha$",
        "scaled attention" : "$\boldsymbol\alpha\nabla\alpha$",
        "lime" : "\textbf{LIME}",
        "fixed-len_var-feat" : "OURS"
    }

    nicer_tasknames = {
        "sst" : "SST",
        "evinf" : "Ev.Inf.",
        "multirc" : "MultiRC",
        "agnews" : "AG"
    }

    new_data_means = {}
    new_with_arrows = {}

    for rationale_type in ["topk"]:

        new_with_arrows = {}
        new_data_means = {}

        for metric in ["f1 macro avg - model labels", "sufficiency", "comprehensiveness"]:

            for task_name in datasets:


                task_name_nicer = nicer_tasknames[task_name]

                new_with_arrows[f"{task_name_nicer}-{metric}-{rationale_type}"] = {}
                new_data_means[f"{task_name_nicer}-{metric}-{rationale_type}"] = {}

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

                for feat_attr in ["deeplift", "lime", "attention", "scaled attention", "ig", "gradients"]:

                    for var_or_fixed in ["fixed", "var"]:

                        if "f1" in metric:

                            new_data_means[f"{task_name_nicer}-{metric}-{rationale_type}"][f"{var_or_fixed}-{mapper[feat_attr]}"] =  data[metric][f"{var_or_fixed}-{feat_attr}"]

                        else:

                            new_data_means[f"{task_name_nicer}-{metric}-{rationale_type}"][f"{var_or_fixed}-{mapper[feat_attr]}"] =  data[f"{var_or_fixed}-{feat_attr}"][metric]["mean"]


                    var = new_data_means[f"{task_name_nicer}-{metric}-{rationale_type}"][f"var-{mapper[feat_attr]}"]
                    fixed = new_data_means[f"{task_name_nicer}-{metric}-{rationale_type}"][f"fixed-{mapper[feat_attr]}"]

                    if "f1" in metric:

                        new_with_arrows[f"{task_name_nicer}-{metric}-{rationale_type}"][mapper[feat_attr]] = "\cellcolor{red!20}|UP|" if var > fixed else "\cellcolor{green!25}|DOWN|"

                    else:

                         new_with_arrows[f"{task_name_nicer}-{metric}-{rationale_type}"][mapper[feat_attr]] = "\cellcolor{green!25}|UP|" if var > fixed else "\cellcolor{red!20}|DOWN|"


    detailed = pd.DataFrame(new_data_means).round(3)
    descriptive = pd.DataFrame(new_with_arrows)

    os.makedirs(f"{save_to_dir}/var_len_table/", exist_ok = True)

    descriptive.to_latex(f"{save_to_dir}/var_len_table/{rationale_type}-this.tex", escape=False)
    
    print(f"** variable-length rationale tables saved in --> {save_to_dir}/var_len_table/ ")
    
    return detailed

## R.I.
def make_tables_for_rationale_length_var_(
    datasets : list = ["sst", "multirc", "agnews", "evinf"],
    metrics_folder : str = "faithfulness_metrics",
    rationale_type: str = "contigious",
    metrics : list = ["f1 macro avg - model labels", "sufficiency", "comprehensiveness"],
    save_to_dir : str = "graphs_and_tables",
    divergence : str= "jsd",
    we_want_sig : bool = False):

    mapper = {
    "gradients" : "$\\boldsymbol{\mathbf{x}\nabla\mathbf{x}}$",
    "ig" : "\textbf{IG}",
    "deeplift" : "\textbf{DeepLift}",
    "attention" : "$\\boldsymbol{\\alpha}$",
    "scaled attention" : "$\\boldsymbol{\\alpha\\nabla\\alpha}$",
    "lime" : "\textbf{LIME}",
    "fixed-len_var-feat" : "OURS"
    }

    nicer_tasknames = {
        "sst" : "SST",
        "evinf" : "Ev.Inf.",
        "multirc" : "MultiRC",
        "agnews" : "AG"
    }

    new_data_means = {}
    new_with_arrows = {}

    for rationale_type in [rationale_type]:

        new_with_arrows = {}
        new_data_means = {}

        for metric in metrics:

            for task_name in datasets:


                task_name_nicer = nicer_tasknames[task_name]

                new_with_arrows[f"{task_name_nicer}-{metric}-{rationale_type}"] = {}
                new_data_means[f"{task_name_nicer}-{metric}-{rationale_type}"] = {}

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
                            f"{rationale_type}-test-faithfulness-metrics.json"
                        )

                    with open(fname, "r") as file : data_sig = json.load(file) 

                    fname = os.path.join(
                        divergence,
                        metrics_folder,
                        task_name,
                        f"{rationale_type}-test-faithfulness-metrics-description.json"
                    )

                with open(fname, "r") as file : data = json.load(file) 

                for feat_attr in ["deeplift", "lime", "attention", "scaled attention", "ig", "gradients"]:

                    temp = {}

                    for var_or_fixed in ["fixed", "var"]:

                        temp[f"{var_or_fixed}-{feat_attr}"] = []  


                        if "f1" in metric:

                            new_data_means[f"{task_name_nicer}-{metric}-{rationale_type}"][f"{var_or_fixed}-{mapper[feat_attr]}"] =  data[metric][f"{var_or_fixed}-{feat_attr}"]

                        else:

                            new_data_means[f"{task_name_nicer}-{metric}-{rationale_type}"][f"{var_or_fixed}-{mapper[feat_attr]}"] =  data[f"{var_or_fixed}-{feat_attr}"][metric]["mean"]

                        if we_want_sig:

                            for annot_id in data_sig:

                                temp[f"{var_or_fixed}-{feat_attr}"].append(
                                    data_sig[annot_id][f"{var_or_fixed}-{feat_attr}"][metric]
                                )


                    if we_want_sig:

                        var = temp[f"var-{feat_attr}"]
                        fixed = temp[f"fixed-{feat_attr}"]

                        pval = wilcoxon(
                            var, 
                            fixed
                        ).pvalue

                        if pval < 0.05:

                            sigi = "^*"

                        else:

                            sigi = ""

                    var = new_data_means[f"{task_name_nicer}-{metric}-{rationale_type}"][f"var-{mapper[feat_attr]}"]
                    fixed = new_data_means[f"{task_name_nicer}-{metric}-{rationale_type}"][f"fixed-{mapper[feat_attr]}"]

                    if we_want_sig:


                        new_with_arrows[f"{task_name_nicer}-{metric}-{rationale_type}"][mapper[feat_attr]] = f"{round(var/fixed,2)}{sigi}"
                    
                    else:

                        new_with_arrows[f"{task_name_nicer}-{metric}-{rationale_type}"][mapper[feat_attr]] = f"{round(var/fixed,2)}"

    detailed = pd.DataFrame(new_data_means).round(3)
    descriptive = pd.DataFrame(new_with_arrows)

    os.makedirs(f"{save_to_dir}/var_len_table/", exist_ok = True)
    
    descriptive.to_csv(f"{save_to_dir}/var_len_table/{rationale_type}-this.csv")

    descriptive.to_latex(f"{save_to_dir}/var_len_table/{rationale_type}-this.tex", escape=False)
    
    print(f"** variable-length rationale tables saved in --> {save_to_dir}/var_len_table/ ")
    
    return