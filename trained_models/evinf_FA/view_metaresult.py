import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

result = np.load('scibert-output_seed-sci5.npy', allow_pickle= True)
#print(result)

df = pd.DataFrame.from_dict(result.item(0)).T
print(df)

pred_labels = []
for pred_label in df['predicted']:
    label = np.argmax(pred_label)
    pred_labels.append(label)

df['pred_labels'] = pred_labels

print(df.columns)
print(df['actual'])
print('accuracy: ', accuracy_score(list(df['actual']), pred_labels))