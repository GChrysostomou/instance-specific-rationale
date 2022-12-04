

import numpy as np


use_topk = True
faithfulness_results = np.load('./posthoc_results/sst/ZEROOUTlimit-faithfulness-scores-detailed.npy', allow_pickle=True).item()
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



descriptor = {}
for feat_attr in {"random", "attention", "scaled attention", "ig", "gradients", "deeplift"}: #"gradientshap", "lime","deepliftshap",
        
    if use_topk: # 0.05, 0.1, 0.2, 0.5]
        sufficiencies_005 = np.asarray([faithfulness_results[k][feat_attr]["sufficiency @ 0.05"] for k in faithfulness_results.keys()])
        comprehensivenesses_005 = np.asarray([faithfulness_results[k][feat_attr]["comprehensiveness @ 0.05"] for k in faithfulness_results.keys()])

        sufficiencies_01 = np.asarray([faithfulness_results[k][feat_attr]["sufficiency @ 0.1"] for k in faithfulness_results.keys()])
        comprehensivenesses_01 = np.asarray([faithfulness_results[k][feat_attr]["comprehensiveness @ 0.1"] for k in faithfulness_results.keys()])

        sufficiencies_02 = np.asarray([faithfulness_results[k][feat_attr]["sufficiency @ 0.2"] for k in faithfulness_results.keys()])
        comprehensivenesses_02 = np.asarray([faithfulness_results[k][feat_attr]["comprehensiveness @ 0.2"] for k in faithfulness_results.keys()])

        sufficiencies_05 = np.asarray([faithfulness_results[k][feat_attr]["sufficiency @ 0.5"] for k in faithfulness_results.keys()])
        comprehensivenesses_05 = np.asarray([faithfulness_results[k][feat_attr]["comprehensiveness @ 0.5"] for k in faithfulness_results.keys()])


        aopc_suff= np.asarray([faithfulness_results[k][feat_attr]["sufficiency aopc"]["mean"] for k in faithfulness_results.keys()])
        aopc_comp = np.asarray([faithfulness_results[k][feat_attr]["comprehensiveness aopc"]["mean"] for k in faithfulness_results.keys()])
        
        descriptor[feat_attr] = {
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


            "AOPC - sufficiency" : {
                "mean" : aopc_suff.mean(),
                "std" : aopc_suff.std()
            },
            "AOPC - comprehensiveness" : {
                "mean" : aopc_comp.mean(),
                "std" : aopc_comp.std()
            }
        }        
        description_fname = "./posthoc_results/sst/" + f"ZEROOUTlimit-faithfulness-scores-description.json"
    else:
        sufficiencies = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency"] for k in faithfulness_results.keys()])
        comprehensivenesses = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness"] for k in faithfulness_results.keys()])
        
        descriptor[feat_attr] = {
            "sufficiency" : {
                "mean" : sufficiencies.mean(),
                "std" : sufficiencies.std()
            },
            "comprehensiveness" : {
                "mean" : comprehensivenesses.mean(),
                "std" : comprehensivenesses.std()
            },
        }
        description_fname = "./posthoc_results/sst/" + f"ZEROOUT-faithfulness-scores-description.json"

    #np.save(detailed_fname, faithfulness_results)
with open(description_fname, 'w') as file:
            json.dump(descriptor,file,indent = 4) 
