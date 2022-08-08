import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = np.load('contigious/dev-rationale_metadata.npy', allow_pickle=True).item(0)
id_list = list(data.keys())

selected_feat_list = []
len_ratio_list = []
rank_type_list = []
for id in id_list:
    selected_feat = data[id]['var-len_var-feat_var-type']['feature attribution name']
    selected_feat_list.append(selected_feat)

    length = data[id]['var-len_var-feat_var-type']['variable rationale ratio']
    len_ratio_list.append(length)

    # rank_type = data[id]['var-len_var-feat_var-type']
    # rank_type_list.append(rank_type)




values, counts = np.unique(selected_feat_list, return_counts=True)
print(values, counts)

mode = max(set(selected_feat_list), key=selected_feat_list.count)
print('the mode feture is :', mode)

sns.distplot( a=len_ratio_list, hist=True, kde=False, rug=False )
plt.show()



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