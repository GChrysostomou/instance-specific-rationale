from pickle import NONE
from re import T
import pandas as pd
import pandas as pd
import json

'''
agnews 
'''



dataset = 'agnews'  #  evinf agnews multirc SST





def generate_csv(dataset, method, std):
    if dataset == 'agnews':
        random_seed = 25
    elif dataset == 'SST':
        random_seed = 10
    elif dataset == 'evinf':
        random_seed = 25
    else:
        print(' no data randomseed')


    if method == NONE:
        if dataset == './evinf': path = './posthoc_results/evinf/topk-faithfulness-scores-average-sci25-description.json'
        else: path = './posthoc_results/' + str(dataset)+'/topk-faithfulness-scores-average-description.json'
    elif method == 'NOISE':
        if dataset == './evinf': path = './posthoc_results/evinf/'+str(method)+'-faithfulness-scores-std_'+str(std)+'-description.json'
        else: path = './posthoc_results/' + str(dataset)+'/NOISE-faithfulness-scores-description'+'-std_'+str(std)+'.json'
    else:
        if dataset == './evinf': path = './posthoc_results/evinf/'+str(method)+'-faithfulness-scores-averages-sci25-description.json'
        else: path = './posthoc_results/' + str(dataset)+'/'+str(method)+'-faithfulness-scores-description.json'


    print(path)
    df = pd.read_json(path, orient ='index')
    print(df)
    df.rename(columns = 
            {'AOPC - sufficiency':'AOPC_sufficiency', 'AOPC - comprehensiveness':'AOPC_comprehensiveness'}, 
            inplace = True)
    sufficiency_mean = []
    comprehensiveness_mean = []


    fea_list = ['random', 'attention', "scaled attention", "gradients", "ig", "gradientshap", "deeplift"]
    for feat in fea_list:
        if method == NONE:
            sufficiency_mean.append(df.AOPC_sufficiency[str(feat)].get('mean'))
            comprehensiveness_mean.append(df.AOPC_comprehensiveness[str(feat)].get('mean'))
        else: # soft true --> no aopc
            sufficiency_mean.append(df.sufficiency[str(feat)].get('mean'))
            comprehensiveness_mean.append(df.comprehensiveness[str(feat)].get('mean'))



    if method != NONE:
         
        random_suff = df.sufficiency['random'].get('mean')
        random_comp = df.comprehensiveness['random'].get('mean')
        
        Suff_ratio = [x / random_suff for x in sufficiency_mean]
        Comp_ratio = [x / random_comp for x in comprehensiveness_mean]

        final_df = pd.DataFrame(list(zip(fea_list, sufficiency_mean, Suff_ratio, comprehensiveness_mean, Comp_ratio)),
                columns =['feature', 'Soft_sufficiency', 'Suff_ratio', 'Soft_comprehensiveness', 'Comp_ratio'])
        if method == 'NOISE': final_df.to_csv(dataset+'/' + str(method) + str(std) +'_faithfulness_result.csv')
        else: final_df.to_csv('./posthoc_results' + dataset+'/' + str(method) +'_faithfulness_result.csv')
        print('saved csv: ', './posthoc_results'+ dataset+'/' + str(method) +'_faithfulness_result.csv')

    else: # not soft, so have aopc
        random_suff = df.AOPC_sufficiency['random'].get('mean')
        random_comp = df.AOPC_comprehensiveness['random'].get('mean')

        Suff_ratio = [x / random_suff for x in sufficiency_mean]
        Comp_ratio = [x / random_comp for x in comprehensiveness_mean]

        final_df = pd.DataFrame(list(zip(fea_list, sufficiency_mean, Suff_ratio, comprehensiveness_mean, Comp_ratio)),
                columns =['feature', ' AOPC_sufficiency', 'Suff_ratio', 'AOPC_comprehensiveness', 'Comp_ratio'])
        final_df.to_csv('./posthoc_results' + dataset+'/faithfulness_result.csv')


# generate_csv(str(dataset), 'NOISE', 1.0)
generate_csv(str(dataset), 'NOISE', 0.5)
generate_csv(str(dataset), 'NOISE', 0.05)
# generate_csv(str(dataset), 'ATTENTION', 1.0)
# generate_csv(str(dataset), 'ZEROOUT', 1.0)
# generate_csv(str(dataset), NONE, 1.0)




