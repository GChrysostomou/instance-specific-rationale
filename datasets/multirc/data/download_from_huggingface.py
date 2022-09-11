from datasets import load_dataset
import pandas as pd
#train = load_dataset('super_glue', 'multirc', split='train')
#test = load_dataset('super_glue', 'multirc', split='test')
# dev = load_dataset('super_glue', 'multirc', split='validation')
# print(dev[0])


df = pd.read_csv('train.csv')
print(df.head())

list = []
for l in df["label"]:
    if str(l) == "False":
        list.append(0)
    elif str(l) == "True": 
        list.append(1)


df['label'] = list
df.to_csv('train.csv')
print(df.head())