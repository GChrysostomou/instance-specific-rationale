#!/bin/bash
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=6-00:00

# set name of job
#SBATCH --job-name=PantryBERT

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

dataset="evinf"
data_dir="datasets/"
model_dir="multilingual_trained_models/"

for seed in 25
do
python finetune_on_ful.py --dataset $dataset \
                          --model_dir $model_dir \
                          --data_dir $data_dir \
                          --seed $seed \
                          --if_multi
done
python finetune_on_ful.py --dataset $dataset \
                          --model_dir $model_dir \
                          --data_dir $data_dir \
                          --seed $seed  \
                          --evaluate_models \
                          --if_multi