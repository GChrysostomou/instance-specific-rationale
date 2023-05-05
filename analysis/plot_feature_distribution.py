import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

dataset = 'evinf'
dataset_path = f'../extracted_rationales/{dataset}/topk/test-rationale_metadata.npy'
data = np.load(dataset_path, allow_pickle=True).item(0)
print(data)

id_list = list(data.keys())

selected_feat_list = []
len_ratio_list = []
rank_type_list = []
for id in id_list:
    selected_feat = data[id]['var-len_var-feat_var-type']['feature attribution name']
    selected_feat_list.append(selected_feat)

    length_ratio = data[id]['var-len_var-feat_var-type']['variable rationale ratio']
    len_ratio_list.append(length_ratio)

    length = data[id]['var-len_var-feat_var-type']['variable rationale length']
    len_list.append(length)

    # rank_type = data[id]['var-len_var-feat_var-type']
    # rank_type_list.append(rank_type)

df = pd.DataFrame(list(zip(id_list, len_ratio_list, len_list, selected_feat_list)),
               columns =['ID', 'length ratio', 'length', 'feature'])


values, counts = np.unique(selected_feat_list, return_counts=True)
percentages = dict(zip(values, counts * 100 / len(selected_feat_list)))
print('features stat: ', values, counts, percentages)

mode = max(set(selected_feat_list), key=selected_feat_list.count)
print('the mode feture is :', mode)

sns.distplot( a=len_ratio_list, hist=True, kde=False, rug=False ).set(title=f'{dataset}_length_ratio_distribution')
plt.show()
plt.savefig(f'{dataset}_length_ratio_distribution.png')

sns.distplot( a=len_list, hist=True, kde=False, rug=False ).set(title=f'{dataset}_length_distribution')
plt.show()
plt.savefig(f'{dataset}_length_distribution.png')

sns.catplot(x="feature", kind="count", palette="ch:.25", data=df)
plt.show()
plt.savefig(f'{dataset}_feature_distribution.png')


# "var-var-len_var-feat_var-type": {
#         "comprehensiveness": {
#             "mean": 0.8624595439371964,
#             "std": 0.30080615632053515
#         },
#         "sufficiency": {
#             "mean": 0.43847272257246206,
#             "std": 0.43110193656380325
#         }


#         "var-scaled attention": {
#         "comprehensiveness": {
#             "mean": 0.7443637638659564,
#             "std": 0.39599188450158435
#         },
#         "sufficiency": {
#             "mean": 0.33490326419689354,
#             "std": 0.40531052006287416
#         }
