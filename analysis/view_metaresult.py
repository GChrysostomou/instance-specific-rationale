from heapq import merge
import pandas as pd
import numpy as np
import json
from sklearn.metrics import accuracy_score
#from src.evaluation.experiments.increasing_feature_scoring import compute_faithfulness_

#### check accuracy and sufficiency...

dataset = 'evinf'
rationale_type_dataset = str(dataset) + '_top'
result = np.load(f'./trained_models/{rationale_type_dataset}/scibert-output_seed-sci15.npy', allow_pickle= True)
meta_top = f'./faithfulness_metrics/{dataset}/topk-test-faithfulness-metrics.json'
meta_conti = f'./faithfulness_metrics/{dataset}/contigious-test-faithfulness-metrics.json'

get_map = pd.read_csv(f'./datasets/{rationale_type_dataset}/data/test.csv')
label_num = get_map['label'].value_counts().to_frame()
label_num = label_num.index.to_list()
feat_num = get_map['feat'].value_counts().to_frame()
label_feat = feat_num.index.to_list()

map = {}
for key in label_num:
    for value in label_feat:
        map[key] = value
        label_feat.remove(value)
        break  


print(map)

with open(meta_top, "r") as file : prediction_data_top = json.load(file)
with open(meta_conti, "r") as file : prediction_data_conti = json.load(file)

df = pd.DataFrame.from_dict(result.item(0)).T
print(df[:5])


#### get accuracy
pred_labels = []
for pred_label in df['predicted']:
    label = np.argmax(pred_label)
    pred_labels.append(label)

df['pred_labels'] = pred_labels

print(df.columns)
print(df['actual'])
print('accuracy: ', accuracy_score(list(df['actual']), pred_labels))


print('-----------')

suff_list_top = []
comp_list_top = []
best_suff_list_top = []
best_comp_list_top = []

for id in prediction_data_top.keys():
    label_num = df.loc[id]['pred_labels']
    label_feat = 'fixed-' + str(map.get(label_num))

    suff_top = prediction_data_top.get(str(id)).get(label_feat).get('sufficiency')
    comp_top = prediction_data_top.get(str(id)).get(label_feat).get('comprehensiveness')
    best_suff_top = prediction_data_top.get(str(id)).get('var-var-len_var-feat').get('sufficiency')
    best_comp_top = prediction_data_top.get(str(id)).get('var-var-len_var-feat').get('comprehensiveness')
    suff_list_top.append(suff_top)
    comp_list_top.append(comp_top)
    best_suff_list_top.append(best_suff_top)
    best_comp_list_top.append(best_comp_top)

print('--- top ----')
print('--- classifier ----suff-->comp')
print(sum(suff_list_top)/len(suff_list_top))
print(sum(comp_list_top)/len(comp_list_top))

# print(sum(best_suff_list_top)/len(best_suff_list_top))
# print(sum(best_comp_list_top)/len(best_comp_list_top))

'''
print(prediction_data_top.get('3253463_0').keys())
dict_keys(['fixed-deeplift', 'fixed-lime', 'fixed-attention', 'fixed-ig', 'fixed-gradients', 'fixed-scaled attention', 'fixed-random', 
'fixed-fixed-len_var-feat', 'fixed-fixed-len_var-feat_var-type', 
              'var-deeplift', 'var-lime', 'var-attention', 'var-ig', 'var-gradients', 'var-scaled attention', 'var-random', 
'var-var-len_var-feat', 'var-var-len_var-feat_var-type']
'''

suff_list_conti = []
comp_list_conti = []
best_suff_list_conti = []
best_comp_list_conti = []

for id in prediction_data_conti.keys():
    label_num = df.loc[id]['pred_labels']
    label_feat = 'fixed-' + str(map.get(label_num))

    suff_conti = prediction_data_conti.get(str(id)).get(label_feat).get('sufficiency')
    comp_conti = prediction_data_conti.get(str(id)).get(label_feat).get('comprehensiveness')
    best_suff_conti = prediction_data_conti.get(str(id)).get('var-var-len_var-feat').get('sufficiency')
    best_comp_conti = prediction_data_conti.get(str(id)).get('var-var-len_var-feat').get('comprehensiveness')
    suff_list_conti.append(suff_conti)
    comp_list_conti.append(comp_conti)
    best_suff_list_conti.append(best_suff_conti)
    best_comp_list_conti.append(best_comp_conti)

print('--- conti ----')
print('--- classifier ----suff-->comp')
print(sum(suff_list_conti)/len(suff_list_conti))
print(sum(comp_list_conti)/len(comp_list_conti))
# print('--- georgy ----')
# print(sum(best_suff_list_conti)/len(best_suff_list_conti))
# print(sum(best_comp_list_conti)/len(best_comp_list_conti))