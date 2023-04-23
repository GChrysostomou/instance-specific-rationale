#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhixue.zhao@sheffield.ac.uk

#SBATCH --time=90:00:00

#SBATCH --job-name=U-ChnSentiCorp


cd /mnt/parscratch/users/cass/BP_MU
export SLURM_EXPORT_ENV=ALL
module load Anaconda3/2022.10
module load CUDA/11.8.0
source activate faith


model_shortname="deberta" #distilbert  mbert  deberta
dataset="ChnSentiCorp"  #["ChnSentiCorp", "ant", "csl", "sst", "evinf", "multirc", "agnews"]


data_dir="datasets/"
model_dir="_trained_models/" 
extracted_rationale_dir="_extracted_rationales/"
evaluation_dir="_faithfulness/"

model_dir="${model_shortname}${model_dir}"
extracted_rationale_dir="${model_shortname}${extracted_rationale_dir}"
evaluation_dir="${model_shortname}${evaluation_dir}"


########## train and predict ###########
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
                          --evaluate_models 
                
                          


python extract_rationales.py --dataset $dataset \
                            --model_dir $model_dir \
                            --data_dir $data_dir \
                            --extracted_rationale_dir $extracted_rationale_dir 
                                    

python evaluate_posthoc.py --dataset $dataset \
                            --data_dir $data_dir \
                            --model_dir $model_dir \
                            --extracted_rationale_dir $extracted_rationale_dir \
                            --evaluation_dir $evaluation_dir
