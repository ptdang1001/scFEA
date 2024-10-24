#!/bin/bash

#SBATCH --job-name=gpu_job            # Job name
#SBATCH --partition=gpu               # Use the GPU partition (modify as needed)
#SBATCH --gpus=1                      # Request 1 GPU
#SBATCH --mem=16G                     # Request 16GB of memory
#SBATCH --time=02:00:00               # Set a 2-hour time limit

python src/main.py \
    --input_dir_path /home/exacloud/gscratch/BDRL/pengtao/Data/scRNA_seq \
    --network_dir_path /home/exacloud/gscratch/BDRL/pengtao/Data/Reaction_Data \
    --output_dir_path /home/exacloud/gscratch/BDRL/pengtao/Results/scFEA \
    --gene_expression_file_name GSE72056_gene569_cell4486.csv.gz \
    --compounds_reactions_file_name M171_V3_connected_cmMat.csv \
    --reactions_genes_file_name M171_V3_connected_reactions_genes.json \
    --experiment_name Flux \
    --n_epoch 50
