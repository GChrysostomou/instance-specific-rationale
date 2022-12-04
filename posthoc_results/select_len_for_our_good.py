import csv
import numpy as np
import json

feat_attr_dict = {"random", "attention", "scaled attention", "ig", "gradients", "deeplift"}



method = 'topk' # from topk ATTENTIONlimit NOISElimit ZEROOUTlimit
rationale_ratios = [0.02, 0.1, 0.2, 0.5] # [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
dataset = 'sst'


def generate_json(dataset, method, rationale_ratios):
    faithfulness_results = np.load(f'./{dataset}/{method}-faithfulness-scores-detailed.npy', allow_pickle=True).item() 
    descriptor = {}
    for feat_attr in ["random", "attention", "scaled attention", "ig", "gradients", "deeplift"]: #"gradientshap", "lime","deepliftshap",
        descriptor[feat_attr] = {}
        aopc_suff = []
        aopc_comp= []

        for ratio in rationale_ratios:

            sufficiencies_k = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ {ratio}"] for k in faithfulness_results.keys()])
            comprehensivenesses_k = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ {ratio}"] for k in faithfulness_results.keys()])

        
            descriptor[feat_attr][f"sufficiency @ {ratio}"] = {"mean" : sufficiencies_k.mean(),
                                                                            "std" : sufficiencies_k.std()
                                                                            }
            descriptor[feat_attr][f"comprehensivenesses @ {ratio}"] = {"mean" : comprehensivenesses_k.mean(),
                                                                            "std" : comprehensivenesses_k.std()
                                                                            }
            
            # RI
            # print(descriptor["random"][f"sufficiency @ {ratio}"]["mean"])
            # print('')
            descriptor[feat_attr][f"sufficiency @ {ratio} to Random RI"] = descriptor[feat_attr][f"sufficiency @ {ratio}"]["mean"]/descriptor["random"][f"sufficiency @ {ratio}"]["mean"]
            descriptor[feat_attr][f"comprehensivenesses @ {ratio} to Random RI"] = descriptor[feat_attr][f"comprehensivenesses @ {ratio}"]["mean"]/descriptor["random"][f"comprehensivenesses @ {ratio}"]["mean"]
                                                                    

            aopc_suff.append(sufficiencies_k)
            aopc_comp.append(comprehensivenesses_k)
        
        descriptor[feat_attr][f"{rationale_ratios} - sufficiency"] = {"mean" : np.array(aopc_suff).mean(), "std" : np.array(aopc_suff).std()}
        descriptor[feat_attr][f"{rationale_ratios} - comprehensiveness"] = { "mean" : np.array(aopc_comp).mean(), "std" : np.array(aopc_comp).std()}
        
        #RI
        descriptor[feat_attr][f"{rationale_ratios} - sufficiency to Random RI"] = descriptor[feat_attr][f"{rationale_ratios} - sufficiency"]["mean"]/descriptor["random"][f"{rationale_ratios} - sufficiency"]["mean"]
        descriptor[feat_attr][f"{rationale_ratios} - comprehensiveness to Random RI"] = descriptor[feat_attr][f"{rationale_ratios} - comprehensiveness"]["mean"]/descriptor["random"][f"{rationale_ratios} - comprehensiveness"]["mean"]

    # description_fname = f"./{dataset}/{method}-{rationale_ratios}-faithfulness-scores-description.json"

    # with open(description_fname, 'w') as file: json.dump(descriptor,file,indent = 4) 
    return descriptor




generate_json(dataset, method, rationale_ratios)