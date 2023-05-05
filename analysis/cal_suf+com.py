import json
import pandas as pd

dataset = 'evinf'
description_path = f'../faithfulness_metrics/{dataset}/topk-test-faithfulness-metrics-description.json'
new_path = f'../faithfulness_metrics/{dataset}/topk-test-faithfulness-metrics-description_new.json'
with open(description_path) as f:
    top_dict = json.load(f)

keys = top_dict.keys()

suff_comp_list1 = []
for key in keys:
    suff_comp = (top_dict[key]['sufficiency']['mean'] + top_dict[key]['comprehensiveness']['mean'])/2
    suff_comp_std = (top_dict[key]['sufficiency']['std'] + top_dict[key]['comprehensiveness']['std'])/2
    top_dict[key]['suff_comp'] = {'mean': suff_comp, 'std': suff_comp_std}
    suff_comp_list1.append(suff_comp)
    print(key)
    print(suff_comp)


# dataset = 'sst'
# description_path = f'../faithfulness_metrics/{dataset}/topk-test-faithfulness-metrics-description.json'
# new_path = f'../faithfulness_metrics/{dataset}/topk-test-faithfulness-metrics-description_new.json'
# with open(description_path) as f:
#     top_dict = json.load(f)

# keys = top_dict.keys()

# suff_comp_list2 = []
# for key in keys:
#     suff_comp = (top_dict[key]['sufficiency']['mean'] + top_dict[key]['comprehensiveness']['mean'])/2
#     suff_comp_std = (top_dict[key]['sufficiency']['std'] + top_dict[key]['comprehensiveness']['std'])/2
#     top_dict[key]['suff_comp'] = {'mean': suff_comp, 'std': suff_comp_std}
#     suff_comp_list2.append(suff_comp)
#     print(key)
#     print(suff_comp)

# df = pd.DataFrame(zip(keys, suff_comp_list1, suff_comp_list2), columns=['feature', 'evinf', 'sst'])
# df.to_csv('suff_comp.csv')

with open(new_path, 'w') as file:
    json.dump(
        top_dict,
        file,
        indent = 4
    )
# 'fixed-deeplift', 'fixed-lime', 'fixed-attention', 'fixed-ig', 'fixed-gradients', 'fixed-scaled attention', 'fixed-random', 
# 'var-deeplift', 'var-lime', 'var-attention', 'var-ig', 'var-gradients', 'var-scaled attention', 'var-random', 

# 'fixed-fixed-len_var-feat', 'fixed-fixed-len_var-feat_var-type', 

# 'var-var-len_var-feat', 'var-var-len_var-feat_var-type'