import json
import numpy as np

faithfulness_results = np.load('topk-faithfulness-scores-detailed.npy', allow_pickle=True).item()


descriptor = {}
    # filling getting averages
for feat_attr in {"random", "attention", "scaled attention", "ig", "gradients", 
        "deeplift"}: #"ig", "gradientshap", , "lime"

    
    sufficiencies_001 = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ 0.01"] for k in faithfulness_results.keys()])
    comprehensivenesses_001 = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ 0.01"] for k in faithfulness_results.keys()])

    sufficiencies_002 = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ 0.02"] for k in faithfulness_results.keys()])
    comprehensivenesses_002 = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ 0.02"] for k in faithfulness_results.keys()])

    sufficiencies_005 = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ 0.05"] for k in faithfulness_results.keys()])
    comprehensivenesses_005 = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ 0.05"] for k in faithfulness_results.keys()])

    sufficiencies_01 = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ 0.1"] for k in faithfulness_results.keys()])
    comprehensivenesses_01 = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ 0.1"] for k in faithfulness_results.keys()])

    sufficiencies_02 = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ 0.2"] for k in faithfulness_results.keys()])
    comprehensivenesses_02 = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ 0.2"] for k in faithfulness_results.keys()])

    sufficiencies_05 = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ 0.5"] for k in faithfulness_results.keys()])
    comprehensivenesses_05 = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ 0.5"] for k in faithfulness_results.keys()])

    sufficiencies_05 = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ 0.5"] for k in faithfulness_results.keys()])
    comprehensivenesses_05 = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ 0.5"] for k in faithfulness_results.keys()])

    sufficiencies_10 = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ 1.0"] for k in faithfulness_results.keys()])
    comprehensivenesses_10 = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ 1.0"] for k in faithfulness_results.keys()])

    aopc_suff= np.asarray([faithfulness_results[k][feat_attr][f"sufficiency aopc"]["mean"] for k in faithfulness_results.keys()])
    aopc_comp = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness aopc"]["mean"] for k in faithfulness_results.keys()])

    
    descriptor[feat_attr] = {
        "sufficiencies @ 0.01" : {
            "mean" : sufficiencies_001.mean(),
            "std" : sufficiencies_001.std()
        },
        "comprehensiveness @ 0.01" : {
            "mean" : comprehensivenesses_001.mean(),
            "std" : comprehensivenesses_001.std()
        },

        "sufficiencies @ 0.02" : {
            "mean" : sufficiencies_002.mean(),
            "std" : sufficiencies_002.std()
        },
        "comprehensiveness @ 0.02" : {
            "mean" : comprehensivenesses_002.mean(),
            "std" : comprehensivenesses_002.std()
        },

        "sufficiencies @ 0.05" : {
            "mean" : sufficiencies_005.mean(),
            "std" : sufficiencies_005.std()
        },
        "comprehensiveness @ 0.05" : {
            "mean" : comprehensivenesses_005.mean(),
            "std" : comprehensivenesses_005.std()
        },


        "sufficiencies @ 0.1" : {
            "mean" : sufficiencies_01.mean(),
            "std" : sufficiencies_01.std()
        },
        "comprehensiveness @ 0.1" : {
            "mean" : comprehensivenesses_01.mean(),
            "std" : comprehensivenesses_01.std()
        },

        
        "sufficiencies @ 0.2" : {
            "mean" : sufficiencies_02.mean(),
            "std" : sufficiencies_02.std()
        },
        "comprehensiveness @ 0.2" : {
            "mean" : comprehensivenesses_02.mean(),
            "std" : comprehensivenesses_02.std()
        },
        

        "sufficiencies @ 0.5" : {
            "mean" : sufficiencies_05.mean(),
            "std" : sufficiencies_05.std()
        },
        "comprehensiveness @ 0.5" : {
            "mean" : comprehensivenesses_05.mean(),
            "std" : comprehensivenesses_05.std()
        },


        "sufficiencies @ 1.0" : {
            "mean" : sufficiencies_10.mean(),
            "std" : sufficiencies_10.std()
        },
        "comprehensiveness @ 1.0" : {
            "mean" : comprehensivenesses_10.mean(),
            "std" : comprehensivenesses_10.std()
        },


        "AOPC - sufficiency" : {
            "mean" : aopc_suff.mean(),
            "std" : aopc_suff.std()
        },
        "AOPC - comprehensiveness" : {
            "mean" : aopc_comp.mean(),
            "std" : aopc_comp.std()
        }
        }        



## save descriptors
fname = f"topk-faithfulness-scores-average-description.json"

with open(fname, 'w') as file:
        json.dump(
            descriptor,
            file,
            indent = 4
        ) 
