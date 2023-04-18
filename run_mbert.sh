#!/bin/bash
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=6-00:00

# set name of job
#SBATCH --job-name=agnews

# set number of GPUs
#SBATCH --gres=gpu:1
#SBATCH --partition=small

#SBATCH --mem=60GB

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=zhixue.zhao@sheffield.ac.uk



# run the application
cd /jmain02/home/J2AD003/txk58/zxz22-txk58/BP-rationales/BP-rationales/
module load python/anaconda3
module load cuda/10.2
source activate ood_faith

dataset="agnews"  #["ant", "csl", "sst", "evinf", "multirc", "agnews"]
data_dir="datasets/"
model_dir="mbert_trained_models/"
extracted_rationale_dir="mbert_extracted_rationales/"
evaluation_dir="mbert_faithfulness/"



# ########### train and predict ###########
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
                            --evaluation_dir $evaluation_dir