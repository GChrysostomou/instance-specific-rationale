

import torch
from regex import B
import pandas as pd
import numpy as np




from random_word import RandomWords
r = RandomWords()

# Return a single random word
a = torch.ones([4,1,4])
a = a.squeeze(1)
print(a.size())
b = 1
c = torch.cat([a, a], dim = 0)
c = torch.cat((c, b), dim = 0)
print(c)


quit()

a = float('-inf') * 2
print(a)
ttt = torch.tensor([[1,2,3,float('-inf'),1],[1,2,3,float('-inf'),1],[1,3,3,5,9]])
print(ttt.size())


ex = ttt.unsqueeze(2)
print(ex.size())
bbb = torch.ones(3,5,768)

a = ex * bbb
print(a.size())
print(' --------- ')
print(ex)
quit()
min_value, min_index = ttt.min(1, keepdim=True)
print(min_index)
ttt[min_index] = 0

print(ttt)
quit()

data = np.load('./posthoc_results/sst/ZEROOUTlimit-faithfulness-scores-detailed.npy', allow_pickle=True).item() 
a = data['test_445']['random']
print(a)
quit()


x = np.array([3, 1, 2])
print(np.argsort(x))
print(np.argsort(-x))



a = [1,2,3,4,5]
b = [1,2,3,4,5]

df = pd.DataFrame(list(zip(a, b)), columns =['annotation_id', 'feat'])
print(df)







# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# from ferret import Benchmark

# name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
# model = AutoModelForSequenceClassification.from_pretrained(name)
# tokenizer = AutoTokenizer.from_pretrained(name)

# bench = Benchmark(model, tokenizer)






# query = "Great movie for a great nap!" 
# scores = bench.score(query) 
# print(scores)

# explanations = bench.explain( query, target=2 ) # "Positive" label 
# bench.show_table(explanations)

# evaluations = bench.evaluate_explanations( explanations, target=2 ) 
# bench.show_evaluation_table(evaluations)


# explanations = bench.explain("You look stunning!", target=1)
# evaluations = bench.evaluate_explanations(explanations, target=1)

# print(evaluations)
# bench.show_evaluation_table(evaluations)




