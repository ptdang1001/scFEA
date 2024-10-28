#!/usr/bin/zsh

python src/main.py \
    --input_dir_path ./inputs \
    --network_dir_path ./inputs \
    --output_dir_path ./outputs \
    --gene_expression_file_name GSE72056_gene569_cell4486.csv.gz \
    --compounds_reactions_file_name M171_V3_connected_cmMat.csv \
    --reactions_genes_file_name M171_V3_connected_reactions_genes.json \
    --experiment_name Flux \
    --n_epoch 50
