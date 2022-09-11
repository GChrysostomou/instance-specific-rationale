from heapq import merge
import pandas as pd
import numpy as np
import json
from sklearn.metrics import accuracy_score
#from src.evaluation.experiments.increasing_feature_scoring import compute_faithfulness_

#### check accuracy and sufficiency...

dataset = 'evinf_top'
result = np.load(f'./trained_models/{dataset}/scibert-output_seed-sci15.npy', allow_pickle= True)
meta = f'./faithfulness_metrics/{dataset}/topk-test-faithfulness-metrics.json'
get_map = pd.read_csv(f'./datasets/{dataset}/data/test.csv')

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

with open(meta, "r") as file : prediction_data = json.load(file)

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

suff_list = []
comp_list = []
best_suff_list = []
best_comp_list = []

for id in prediction_data.keys():
    label_num = df.loc[id]['pred_labels']
    label_feat = 'fixed-' + str(map.get(label_num))

    suff = prediction_data.get(str(id)).get(label_feat).get('sufficiency')
    comp = prediction_data.get(str(id)).get(label_feat).get('comprehensiveness')
    best_suff = prediction_data.get(str(id)).get('var-var-len_var-feat').get('sufficiency')
    best_comp = prediction_data.get(str(id)).get('var-var-len_var-feat').get('comprehensiveness')
    suff_list.append(suff)
    comp_list.append(comp)
    best_suff_list.append(best_suff)
    best_comp_list.append(best_comp)


print(sum(suff_list)/len(suff_list))
print(sum(comp_list)/len(comp_list))
print(sum(best_suff_list)/len(best_suff_list))
print(sum(best_comp_list)/len(best_comp_list))

print(prediction_data.get('3253463_0').get('fixed-deeplift').get('comprehensiveness'))


