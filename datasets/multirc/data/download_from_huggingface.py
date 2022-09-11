from datasets import load_dataset

#train = load_dataset('super_glue', 'multirc', split='train')
#test = load_dataset('super_glue', 'multirc', split='test')
# dev = load_dataset('super_glue', 'multirc', split='validation')
# print(dev[0])
import panda as pd

df = pd.read_csv('dev.csv')
df["label"] = df["label"].map({"false": 0, "true": 1})
df.to_csv('de.csv')