Repository for the paper *"Incorporating Attribution Importance for Improving Faithfulness Metrics"* to appear at ACL 2023.

We promise we will tidy up the repo soon : >


## Prerequisites

Install necessary packages by using the files  [conda_reqs.txt](https://github.com/GChrysostomou/instance-specific-rationale/blob/main/conda_reqs.txt) and  [pip_reqs.txt](https://github.com/GChrysostomou/instance-specific-rationale/blob/main/pip_reqs.txt)  

```
conda create --name flexi --file  conda_reqs.txt
conda activate flexi
pip install -r pip_reqs.txt
python -m spacy download en_core_web_sm
```

## Downloading Task Data
You can run the jupyter notebooks found under tasks/*task_name*/\*ipynb to generate a filtered, processed *csv* file and a pickle file used for trainining the models.


## Training the models

```
dataset="evinf"
data_dir="datasets/"
model_dir="trained_models/"

for seed in 5 10 15 
do
python finetune_on_ful.py --dataset $dataset \
                          --model_dir $model_dir \
                          --data_dir $data_dir \
                          --seed $seed 
done
python finetune_on_ful.py --dataset $dataset \
                          --model_dir $model_dir \
                          --data_dir $data_dir \
                          --seed $seed  \
                          --evaluate_models 
```

## Extracting Rationales (Fixed or Instance-Specific - For all parameters)

```
divergence="jsd"
extracted_rationale_dir="extracted_rationales/"

python extract_rationales.py --dataset $dataset  \
                             --model_dir $model_dir \
                             --data_dir $data_dir \
                             --extracted_rationale_dir $extracted_rationale_dir \
                             --extract_double \
                             --divergence $divergence
```
```--extract_double``` is optional and is used to double $N$.

## Evaluating Faithfulness
```
extracted_rationale_dir="extracted_rationales/"
evaluation_dir="faithfulness_metrics/"

python evaluate_masked.py --dataset $dataset \
                          --model_dir $model_dir \
                          --extracted_rationale_dir $extracted_rationale_dir \
                          --data_dir $data_dir \
                          --evaluation_dir $evaluation_dir\
                          --thresholder $thresh
```


## Summarising results

Following the evaluation you can use [src/generate_results/recreate_paper.py](https://github.com/GChrysostomou/instance-specific-rationale/blob/main/src/generate_results/recreate_paper.py) to recreate the tables and figures seen in the paper. 
