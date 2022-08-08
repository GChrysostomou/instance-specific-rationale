from transformers import AutoModelForSequenceClassification, AutoTokenizer
from ferret import Benchmark

name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(name)
tokenizer = AutoTokenizer.from_pretrained(name)

bench = Benchmark(model, tokenizer)






query = "Great movie for a great nap!" 
scores = bench.score(query) 
print(scores)

explanations = bench.explain( query, target=2 ) # "Positive" label 
bench.show_table(explanations)

evaluations = bench.evaluate_explanations( explanations, target=2 ) 
bench.show_evaluation_table(evaluations)


# explanations = bench.explain("You look stunning!", target=1)
# evaluations = bench.evaluate_explanations(explanations, target=1)

# print(evaluations)
# bench.show_evaluation_table(evaluations)




