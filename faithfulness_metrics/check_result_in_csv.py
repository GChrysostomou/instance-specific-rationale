import pandas as pd
import json

dataset = './agnews'
path = str(dataset)+'/topk-test-faithfulness-metrics-description.json'

df = pd.read_json(path, orient ='index')
print(df)
sufficiency_mean = []
comprehensiveness_mean = []
for i in range(7):
    sufficiency_mean.append(df.sufficiency[i].get('mean'))
    comprehensiveness_mean.append(df.comprehensiveness[i].get('mean'))
print(sufficiency_mean)
df[sufficiency_mean] = sufficiency_mean
df[comprehensiveness_mean] = comprehensiveness_mean

feature = ['Deeplift', 'Lime','Attention','IG','Gradients','Scaled Attention','Random']
df = pd.DataFrame(list(zip(feature, sufficiency_mean, comprehensiveness_mean)),
               columns =['feature', 'sufficiency', 'comprehensiveness'])
df.to_csv(dataset+'/faithfulness_result.csv')

