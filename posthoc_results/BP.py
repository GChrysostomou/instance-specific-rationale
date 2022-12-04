

import numpy as np
import json

rationale_ratios = [0.02, 0.1, 0.2, 0.5] # [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
faithfulness_results = np.load('./sst/topk-faithfulness-scores-detailed.npy', allow_pickle=True).item() 
idsample = list(faithfulness_results.keys())[3]


print('')
print('  faithfulness_results[idsample].keys()  ')
print(faithfulness_results[idsample]['ig'])


print('')
for key in faithfulness_results[idsample].keys():
    print(key)
    print(faithfulness_results[idsample][key])
    print('  ')



print(faithfulness_results[idsample]['ig'])


print(' building description --->')
descriptor = {}
for feat_attr in {"random", "attention", "scaled attention", "ig", "gradients", "deeplift"}: #"gradientshap", "lime","deepliftshap",
    print('----  ', feat_attr, '  -------------' )
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

        aopc_suff.append(sufficiencies_k)
        aopc_comp.append(comprehensivenesses_k)
    
    descriptor[feat_attr][f"{rationale_ratios} - sufficiency"] = {"mean" : np.array(aopc_suff).mean(),
                                                        "std" : np.array(aopc_suff).std()},
    descriptor[feat_attr][f"{rationale_ratios} - comprehensiveness"] = { "mean" : np.array(aopc_comp).mean(),
                                                             "std" : np.array(aopc_comp).std()}
                             


description_fname = "./sst/" + f"topk-{rationale_ratios}-faithfulness-scores-description.json"

    #np.save(detailed_fname, faithfulness_results)
print(descriptor)

with open(description_fname, 'w') as file: json.dump(descriptor,file,indent = 4) 
