import pandas as pd
import numpy as np
import json
from sklearn.metrics import accuracy_score
#from src.evaluation.experiments.increasing_feature_scoring import compute_faithfulness_

#### check accuracy and sufficiency...

dataset = 'evinf'
result = np.load(f'../trained_models/{dataset}_FA/scibert-output_seed-sci10.npy', allow_pickle= True)
meta = f'../faithfulness_metrics/{dataset}/topk-test-faithfulness-metrics.json'
get_map = pd.read_csv(f'../datasets/{dataset}_FA/data/test.csv')
label_num = get_map['label'].value_counts().to_frame()
label_num['number'] = label_num.index
feat_num = get_map['feat'].value_counts().to_frame()
feat_num['number'] = feat_num.index

merge = 

print(label_num)
print(feat_num)
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
print(prediction_data.get('3253463_0').get('fixed-deeplift').get('comprehensiveness'))


