# -*-coding:utf-8-*-


# built-in library
import os, json, warnings
from numba import njit
from datetime import datetime


# third-party library
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# global variables
SEP_SIGN = "*" * 100
warnings.filterwarnings("ignore")


def init_output_dir_path(args):
    reaction_network_name = args.compounds_reactions_file_name.split("_cmMat.csv")[0]
    print("Reaction Network Name:{0}".format(reaction_network_name))
    print(SEP_SIGN)
    # Get the current timestamp
    current_timestamp = datetime.now()
    # Format the timestamp
    formatted_timestamp = current_timestamp.strftime("%Y%m%d%H%M%S")
    data_file_name = args.gene_expression_file_name.split(".csv")[0]
    folder_name = f"{data_file_name}-{reaction_network_name}-{args.experiment_name}_{formatted_timestamp}"
    output_dir_path = os.path.join(args.output_dir_path, folder_name)
    # if folder already exists, add a number to the folder name
    if os.path.exists(output_dir_path):
        random_number = np.random.randint(1, 999)
        folder_name = (
            f"{data_file_name}-{reaction_network_name}-{args.experiment_name}_"
            f"{formatted_timestamp}_{str(random_number).zfill(3)}"
        )
        output_dir_path = os.path.join(args.output_dir_path, folder_name)
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    return output_dir_path, folder_name


def load_gene_expression(args):
    read_path = os.path.join(args.input_dir_path, args.gene_expression_file_name)
    gene_expression = None

    if read_path.endswith(".csv.gz"):
        gene_expression = pd.read_csv(read_path, index_col=0, compression="gzip")
    elif read_path.endswith(".csv"):
        gene_expression = pd.read_csv(read_path, index_col=0)
    else:
        print("Wrong Gene Expression File Name!")
        return False

    # replace the nan with zero
    gene_expression = gene_expression.fillna(0.0)

    # remove the rows which are all zero
    gene_expression = gene_expression.loc[~(gene_expression == 0).all(axis=1), :]

    # remove the cols which are all zero
    gene_expression = gene_expression.loc[:, ~(gene_expression == 0).all(axis=0)]

    # remove duplicated rows
    gene_expression = gene_expression[~gene_expression.index.duplicated(keep="first")]

    # remove duplicated cols
    gene_expression = gene_expression.loc[
        :, ~gene_expression.columns.duplicated(keep="first")
    ]

    gene_expression = gene_expression.T

    print(SEP_SIGN)
    # choose 5 random row index
    n_rdm = 5
    rdm_row_idx = np.random.choice(gene_expression.index, n_rdm)
    # choose 5 random col index
    rdm_col_idx = np.random.choice(gene_expression.columns, n_rdm)

    # print the gene expression data, the random 5 rows and 5 cols
    print("Gene Expression Data shape:{0}".format(gene_expression.shape))
    print(
        "Gene Expression Data sample:{0}".format(
            gene_expression.loc[rdm_row_idx, rdm_col_idx]
        )
    )

    print(SEP_SIGN)

    return gene_expression


def load_compounds_reactions(args):
    read_path = os.path.join(args.network_dir_path, args.compounds_reactions_file_name)
    compounds_reactions = pd.read_csv(read_path, index_col=0)

    compounds_reactions.index = compounds_reactions.index.map(lambda x: str(x))
    compounds_reactions = compounds_reactions.astype(int)

    """
    print(SEP_SIGN)
    print("\nCompounds:{0}\n".format(compounds_reactions.index.values))
    print("\nReactions:{0}\n".format(compounds_reactions.columns.values))
    print("\nCompounds_Reactions shape:{0}\n".format(compounds_reactions.shape))
    print("\n compounds_Reactions sample:\n {0} \n".format(compounds_reactions))
    print(SEP_SIGN)
    """
    return compounds_reactions


def load_reactions_genes(args):

    read_path = os.path.join(args.network_dir_path, args.reactions_genes_file_name)
    # Opening JSON file
    f = open(read_path)
    # returns JSON object as
    # a dictionary
    reactions_genes = json.load(f)
    # Closing file
    f.close()

    """
    print(SEP_SIGN)
    print("\n Reactions and contained genes:\n {0} \n".format(reactions_genes))
    print(SEP_SIGN)
    """

    return reactions_genes


def remove_allZero_rowAndCol(factors_nodes):
    # remove all zero rows and columns
    factors_nodes = factors_nodes.loc[~(factors_nodes == 0).all(axis=1), :]
    factors_nodes = factors_nodes.loc[:, ~(factors_nodes == 0).all(axis=0)]
    return factors_nodes


def remove_margin_compounds(factors_nodes):
    n_factors, _ = factors_nodes.shape
    keep_idx = []
    print(SEP_SIGN)
    for i in range(n_factors):
        if (factors_nodes.iloc[i, :] >= 0).all() or (
            factors_nodes.iloc[i, :] <= 0
        ).all():
            # print("Remove Compound:{0}".format(factors_nodes.index.values[i]))
            continue
        else:
            keep_idx.append(i)
    factors_nodes = factors_nodes.iloc[keep_idx, :]
    factors_nodes = remove_allZero_rowAndCol(factors_nodes)
    # print(SEP_SIGN)
    # print(SEP_SIGN)
    # print("\n compounds_modules sample:\n {0} \n".format(factors_nodes))
    # print(SEP_SIGN)
    return factors_nodes


def get_data_with_intersection_gene(gene_expression, reactions_genes):
    all_genes_in_gene_expression = set(gene_expression.columns.values.tolist())
    all_genes_in_reactions = []
    for _, genes in reactions_genes.items():
        all_genes_in_reactions.extend(genes)
    all_genes_in_reactions = set(all_genes_in_reactions)

    intersection_genes = all_genes_in_gene_expression.intersection(
        all_genes_in_reactions
    )

    if len(intersection_genes) == 0:
        return [], []

    reactions_genes_new = {}
    print("Current Reaction - Intersection Genes")
    for reaction_i, genes in reactions_genes.items():
        cur_genes_intersection = None
        cur_genes_intersection = set(genes).intersection(intersection_genes)
        print(f"{reaction_i} - {list(cur_genes_intersection)}")
        if len(cur_genes_intersection) != 0:
            reactions_genes_new[reaction_i] = list(cur_genes_intersection)

    return gene_expression[list(intersection_genes)], reactions_genes_new


def data_pre_processing(gene_expression, reactions_genes, compounds_reactions):
    # get the data with intersection genes
    gene_expression, reactions_genes = get_data_with_intersection_gene(
        gene_expression, reactions_genes
    )

    # if there is no intersection genes, just return
    if len(gene_expression) == 0:
        # print("\n No Intersection of Genes between Data and Reactions! \n")
        return [], [], []

    # for the compounds_reactions adj matrix
    # remove outside compounds and reactions
    compounds_reactions = remove_margin_compounds(compounds_reactions)
    # remove the all zero rows
    # remove the all zero columns
    compounds_reactions = remove_allZero_rowAndCol(compounds_reactions)

    # get the intersection compounds and reactions
    for reaction_i in compounds_reactions.columns.values:
        if reaction_i not in reactions_genes.keys():
            reactions_genes[reaction_i] = []

    # if there is no intersection genes, just return
    if len(gene_expression) == 0:
        # print("\n No Intersection of Genes between Data and Reactions! \n")
        return [], [], []

    return gene_expression, reactions_genes, compounds_reactions

def log_and_min_max_normalize(df):
    log_df = np.log1p(df)
    min_vals = log_df.min(axis=0)
    max_vals = log_df.max(axis=0)
    normalized_df = (log_df - min_vals) / (max_vals - min_vals + 1e-8)
    return normalized_df


def z_score_normalization(data):
    """Apply Z-score normalization to each column."""
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)

    # Avoid division by zero for constant columns
    stds[stds == 0] = 1
    normalized_data = (data - means) / stds
    return normalized_data


@njit
def fill_zeros_with_positive_gaussian(data):
    for i in range(data.shape[1]):
        col = data[:, i]

        # get non 0 values
        non_zero_values = col[col != 0]

        if non_zero_values.size > 0:
            # get mean and std
            mean_value = non_zero_values.mean()
            std_value = non_zero_values.std()

            # replace 0 with positive gaussian
            for j in range(col.size):
                if col[j] == 0:
                    # get a positive gaussian sample
                    sample = -1
                    while sample <= 0:
                        sample = np.random.normal(mean_value, std_value)
                    col[j] = sample
    return data


@njit
def fill_zeros_with_mean(data):
    """Fill zero values with the mean of non-zero values in each column."""
    for i in range(data.shape[1]):
        col = data[:, i]
        non_zero_values = col[col != 0]

        if non_zero_values.size > 0:
            mean_value = non_zero_values.mean()
            for j in range(col.size):
                if col[j] == 0:
                    col[j] = mean_value
    return data


def normalize_gene_expression(gene_expression, reactions_genes):
    reactions_geneExpressionMean = {}
    reactions_gene_expression_normalized = {}

    for reaction, genes in reactions_genes.items():
        if len(genes) == 0:
            continue
        cur_data = None
        cur_data = gene_expression.loc[:, genes].values
        reactions_geneExpressionMean[reaction] = cur_data.mean(axis=1)
        cur_data = log_and_min_max_normalize(cur_data)
        cur_data = fill_zeros_with_positive_gaussian(cur_data)
        reactions_gene_expression_normalized[reaction] = cur_data

    return reactions_gene_expression_normalized, reactions_geneExpressionMean


class CombinedDataset(Dataset):

    def __init__(self, reactions_gene_expression):
        self.reactions_gene_expression = reactions_gene_expression
        self.num_samples = next(iter(reactions_gene_expression.values())).shape[0]

    def __len__(self):
        # Assuming all datasets have the same length
        return self.num_samples

    def __getitem__(self, idx):
        batch = {
            reaction_i: torch.tensor(gene_expression_np[idx], dtype=torch.float32)
            for reaction_i, gene_expression_np in self.reactions_gene_expression.items()
        }
        return batch


def split_data(reactions_gene_expression, test_size=0.2):
    train_data = {}
    test_data = {}
    for reaction_i, gene_expression_np in reactions_gene_expression.items():
        X_train, X_test = train_test_split(gene_expression_np, test_size=test_size)
        train_data[reaction_i] = X_train
        test_data[reaction_i] = X_test
    return train_data, test_data
