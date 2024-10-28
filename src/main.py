# -*-coding:utf-8-*-

# built-in library
import os, argparse, warnings

# Third-party library
import pandas as pd
import numpy as np

# import pysnooper
import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

# my library
from data_interface import load_gene_expression
from data_interface import load_compounds_reactions
from data_interface import load_reactions_genes
from data_interface import data_pre_processing
from data_interface import init_output_dir_path
from data_interface import normalize_gene_expression
from data_interface import split_data
from data_interface import CombinedDataset


from model_interface import DynamicRegressionModel
from model_interface import MultiModelContainer

from MPO.mpo import mpo

from utils import plot_loss_curves
from utils import get_imbalanceLoss


# global variables
SEP_SIGN = "*" * 100
warnings.filterwarnings("ignore")
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")


# @pysnooper.snoop()
def train_validation_inference(
    args,
    reactions_gene_expression_normalized,
    reactions_geneExpressionMean,
    compounds_reactions_df,
    reactions_genes,
):

    # set the random seed
    # L.seed_everything(args.seed)
    
    # define a group of models, # of models = # of reactions
    models = {
        reaction_i: (
            DynamicRegressionModel(input_dim=len(reactions_genes[reaction_i]))
            if len(reactions_genes[reaction_i]) > 0
            else None
        )
        for reaction_i in compounds_reactions_df.columns.tolist()
    }

    # define the multi model container
    multi_model_container = MultiModelContainer(
        models,
        compounds_reactions_df.values,
        compounds_reactions_df.columns.tolist(),
    )

    # split the data into train and validation
    train_datasets, val_datasets = split_data(
        reactions_gene_expression_normalized, test_size=0.3
    )
    train_datasets = CombinedDataset(train_datasets)
    val_datasets = CombinedDataset(val_datasets)
    full_datasets = CombinedDataset(reactions_gene_expression_normalized)
    train_datasets = DataLoader(
        train_datasets,
        batch_size=args.batch_size,
        num_workers=os.cpu_count(),
        shuffle=True,
        pin_memory=True,
    )
    val_datasets = DataLoader(
        val_datasets,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=os.cpu_count(),
    )
    full_datasets = DataLoader(
        full_datasets,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    # create the folder to save the model
    model_save_path = os.path.join(args.output_dir_path, "model_training_log")
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    # monitor the validation loss and save the best model
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_save_path,
        monitor="val_total_loss",
        filename="best-model-{epoch:03d}-{val_loss:.5f}",
        save_top_k=1,
        mode="min",
        save_weights_only=True,
    )

    # early stop
    early_stop_callback = EarlyStopping(
        monitor="val_total_loss", patience=10, verbose=False, mode="min"
    )

    gpu_option = 'auto' if torch.cuda.is_available() else 'cpu'
    # define the trainer
    trainer = Trainer(
        default_root_dir=model_save_path,
        max_epochs=args.n_epoch,
        accelerator=gpu_option,
        devices="auto",
        strategy="auto",
        callbacks=[checkpoint_callback, early_stop_callback],
    )

    print("\nStart training scFEA models ... \n")
    trainer.fit(multi_model_container, train_datasets, val_datasets)
    print("\nFinish training scFEA models ... \n")

    best_model_path = checkpoint_callback.best_model_path
    print(f"The best model weights is saved at: \n {best_model_path} \n")

    multi_model_container.load_state_dict(
        torch.load(best_model_path, weights_only=True)["state_dict"],
    )

    print("\nStart predicting flux ... \n")
    test_trainer = Trainer(
        default_root_dir=model_save_path,
        max_epochs=1,
        accelerator=gpu_option,
        devices=1,
        strategy="auto",
    )

    test_trainer.test(model=multi_model_container, dataloaders=full_datasets)
    samples_reactions_scfea = multi_model_container.predictions
    samples_reactions_scfea = pd.DataFrame(
        samples_reactions_scfea, columns=compounds_reactions_df.columns
    )

    # run mpo

    print("\nStart Running MPO ... \n")
    samples_reactions_scfea_mpo = mpo(
        compounds_modules=compounds_reactions_df,
        samples_modules_input=samples_reactions_scfea,
        main_branch=[],
        samples_mean=None,
        args=args,
    )

    imbalance_loss_scfea = get_imbalanceLoss(
        compounds_reactions_df, samples_reactions_scfea
    )
    imbalance_loss_mpo = get_imbalanceLoss(
        compounds_reactions_df, samples_reactions_scfea_mpo
    )
    cv_loss_scfea = samples_reactions_scfea.std(axis=0) / samples_reactions_scfea.mean(
        axis=0
    )
    cv_loss_scfea = cv_loss_scfea.mean()
    cv_loss_mpo = samples_reactions_scfea_mpo.std(
        axis=0
    ) / samples_reactions_scfea_mpo.mean(axis=0)
    cv_loss_mpo = cv_loss_mpo.mean()
    mpo_loss = {"imbalance_loss": imbalance_loss_mpo, "cv_loss": cv_loss_mpo}
    scfea_loss = {"imbalance_loss": imbalance_loss_scfea, "cv_loss": cv_loss_scfea}
    if mpo_loss["imbalance_loss"] > scfea_loss["imbalance_loss"]:
        samples_reactions_scfea_mpo = (
            samples_reactions_scfea + samples_reactions_scfea_mpo
        ) / 2.0
        mpo_loss["cv_loss"] = (mpo_loss["cv_loss"] + scfea_loss["cv_loss"]) / 2.0
        mpo_loss["imbalance_loss"] = (
            mpo_loss["imbalance_loss"] + scfea_loss["imbalance_loss"]
        ) / 2.0

    print(f"scFEA Imbalance Loss: {imbalance_loss_scfea}")
    print(f"MPO Imbalance Loss: {imbalance_loss_mpo}")
    print(f"scFEA CV Loss: {cv_loss_scfea}")
    print(f"MPO CV Loss: {cv_loss_mpo}")

    # plot the loss curves
    plot_loss_curves(args, trainer.logger.log_dir, mpo_loss, scfea_loss)

    return samples_reactions_scfea, samples_reactions_scfea_mpo


# @pysnooper.snoop()
def main(args):
    # print the input parameters
    print(f"{SEP_SIGN} \nCurrent Input parameters:\n{args}\n {SEP_SIGN}")
    print(f"Current CPU cores: {os.cpu_count()}")
    print(f"Current GPU devices: {torch.cuda.device_count()}")

    # load gene expression data
    # geneExpression is the gene expression data,
    # cols:=samples/cells, rows:=genes,
    # but the data will be transposed to rows:=samples/cells,
    # cols:=genes automatically
    gene_expression_data = load_gene_expression(args)

    # load the reactions and the contained genes, stored in a json file
    reactions_genes = load_reactions_genes(args)

    # load the compounds and the reactions data, it is an adj matrix
    # compouns_reactions is the adj matrix of the factor graph (reaction graph)
    # rows:=compounds, columns:=reactions, entries are 0,1,-1
    compounds_reactions_df = load_compounds_reactions(args)

    # data pre-processing, remove the genes which are not in the reactions_genes
    # and remove the reactions which are not in the compounds_reactions
    # and remove the compounds which are not in the compounds_reactions
    # and remove the samples which are not in the gene_expression_data
    gene_expression_data, reactions_genes, compounds_reactions_df = data_pre_processing(
        gene_expression_data, reactions_genes, compounds_reactions_df
    )
    print(f"Compounds Reactions ADJ Matrix: \n{compounds_reactions_df}\n")

    if len(gene_expression_data) == 0:
        print("\nNo Intersection of Genes between Data and (reactions)Reactions! \n")
        return False

    # normalize the gene expression data
    # return a dictionary, key:=reaction, value:=normalized gene expression data
    (reactions_gene_expression_normalized, reactions_geneExpressionMean) = (
        normalize_gene_expression(gene_expression_data, reactions_genes)
    )

    # initialize the output dir path
    # prepare the input and output dir
    args.output_dir_path, _ = init_output_dir_path(args)

    # sys.exit(1)

    samples_reactions_scfea, samples_reactions_scfea_mpo = train_validation_inference(
        args,
        reactions_gene_expression_normalized,
        reactions_geneExpressionMean,
        compounds_reactions_df,
        reactions_genes,
    )
    samples_reactions_scfea.index = gene_expression_data.index
    samples_reactions_scfea_mpo.index = gene_expression_data.index
    print(f"Flux Result Shape: {samples_reactions_scfea.shape}")
    # randomly pick 10 rows and 10 columns
    rdm_row_idxs = np.random.choice(samples_reactions_scfea.shape[0], 10)
    rdm_col_idxs = np.random.choice(samples_reactions_scfea.shape[1], 10)
    # print the 10 rows and 10 columns, and print the 10 rows and 10 columns of scFEA and MPO
    print(
        f"scFEA Reslut Sample: \n{samples_reactions_scfea.iloc[rdm_row_idxs, rdm_col_idxs]}\n"
    )
    print(
        f"scFEA -> MPO Reslut Sample: \n{samples_reactions_scfea_mpo.iloc[rdm_row_idxs, rdm_col_idxs]}\n"
    )

    # save the scFEA results
    flux_res_save_path = os.path.join(args.output_dir_path, "flux_scfea.csv")
    samples_reactions_scfea.to_csv(
        flux_res_save_path,
        index=True,
        header=True,
    )

    # save the scFEA MPO results
    flux_res_save_path = os.path.join(args.output_dir_path, "flux_scfea_mpo.csv")
    samples_reactions_scfea_mpo.to_csv(
        flux_res_save_path,
        index=True,
        header=True,
    )

    print(f"\nscFEA & MPO results are saved at: \n{args.output_dir_path}\n")

    return True


def parse_arguments(parser):
    # global parameters
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--input_dir_path", type=str, default="./inputs/", help="The inputs directory."
    )
    parser.add_argument(
        "--network_dir_path",
        type=str,
        default="./inputs/",
        help="The inputs directory.",
    )
    parser.add_argument(
        "--output_dir_path",
        type=str,
        default="./outputs/",
        help="The outputs directory, you can find all outputs in this directory.",
    )
    parser.add_argument(
        "--gene_expression_file_name",
        type=str,
        default="NA",
        help="The scRNA-seq file name.",
    )
    parser.add_argument(
        "--compounds_reactions_file_name",
        type=str,
        default="NA",
        help="The table describes relationship between compounds and reactions. Each row is an intermediate metabolite and each column is metabolic reaction.",
    )
    parser.add_argument(
        "--reactions_genes_file_name",
        type=str,
        default="NA",
        help="The json file contains genes for each reaction. We provide human and mouse two models in scFEA.",
    )

    parser.add_argument("--experiment_name", type=str, default="flux")
    parser.add_argument("--reaction_network_name", type=str, default="NA")

    # parameters for scFEA
    parser.add_argument(
        "--n_epoch",
        type=int,
        default=100,
        help="User defined Epoch for scFEA training.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1024, help="Batch size, scfea."
    )
    # parameters for bp_balance
    parser.add_argument(
        "--n_epoch_mpo",
        type=int,
        default=64,
        help="User defined Epoch for Message Passing Optimizer.",
    )
    parser.add_argument(
        "--delta", type=float, default=0.001, help="delta for the stopping criterion"
    )
    parser.add_argument(
        "--beta_1", type=float, default=0.4, help="beta_1 for the update step"
    )
    parser.add_argument(
        "--beta_2", type=float, default=0.5, help="beta_2 for main branch"
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # mp.set_start_method('spawn',force=True)
    parser = argparse.ArgumentParser(description="scFEA")

    # global args
    args = parse_arguments(parser)

    main(args)
