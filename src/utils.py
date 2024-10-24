import os, sys

# import pysnooper
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool


# @pysnooper.snoop()
def plot_loss_curves(args, loss_file_path, mpo_loss, scfea_loss):
    print("Plotting the loss curves...")
    print(f"\nReading loss logs from: {loss_file_path}/metrics.csv \n")

    epoch_metrics = pd.read_csv(os.path.join(loss_file_path, "metrics.csv"))

    # Epoch from 1
    epoch_metrics["epoch"] = epoch_metrics["epoch"].dropna() + 1
    epochs = epoch_metrics["epoch"].unique()

    # get the loss columns
    train_loss_columns = [
        col
        for col in epoch_metrics.columns
        if col.startswith("train_") and "loss" in col
    ]
    val_loss_columns = [
        col for col in epoch_metrics.columns if col.startswith("val_") and "loss" in col
    ]

    val_losses_df = epoch_metrics[["epoch"] + val_loss_columns]
    val_losses_df = val_losses_df[val_losses_df["epoch"] != 1]

    max_val_loss = -float("inf")
    max_val_loss_epoch = None
    for col in val_loss_columns:
        current_max = val_losses_df[col].max()
        if current_max > max_val_loss:
            max_val_loss = current_max
            max_val_loss_epoch = val_losses_df.loc[
                val_losses_df[col] == current_max, "epoch"
            ].values[0]

    # print(f"Maximum validation loss {max_val_loss} found in epoch {int(max_val_loss_epoch)}")

    epoch_metrics = epoch_metrics[epoch_metrics["epoch"] != max_val_loss_epoch]

    epochs = [epoch for epoch in epochs if epoch != max_val_loss_epoch]

    train_losses = {}
    for loss_col in train_loss_columns:
        loss_values = epoch_metrics[["epoch", loss_col]].dropna()
        train_losses[loss_col] = loss_values.groupby("epoch")[loss_col].mean()

    val_losses = {}
    for loss_col in val_loss_columns:
        loss_values = epoch_metrics[["epoch", loss_col]].dropna()
        val_losses[loss_col] = loss_values.groupby("epoch")[loss_col].mean()

    plt.figure(figsize=(10, 6))

    train_color = "blue"
    val_color = "green"

    train_markers = ["o", "v", "^", "<", ">"]
    val_markers = ["s", "P", "*", "X", "D"]

    for i, (loss_col, loss_values) in enumerate(train_losses.items()):
        if loss_col == "train_total_loss":
            plt.plot(
                loss_values.index,
                loss_values.values,
                label=loss_col,
                color=train_color,
                marker=train_markers[i % len(train_markers)],
                linewidth=3.0,
            )  #
        else:
            plt.plot(
                loss_values.index,
                loss_values.values,
                label=loss_col,
                color=train_color,
                marker=train_markers[i % len(train_markers)],
                linewidth=1.0,
                alpha=0.25,
            )  #

    #
    for i, (loss_col, loss_values) in enumerate(val_losses.items()):
        if loss_col == "val_total_loss":
            plt.plot(
                loss_values.index,
                loss_values.values,
                label=loss_col,
                color=val_color,
                marker=val_markers[i % len(val_markers)],
                linewidth=3.0,
            )  #
        else:
            plt.plot(
                loss_values.index,
                loss_values.values,
                label=loss_col,
                color=val_color,
                marker=val_markers[i % len(val_markers)],
                linewidth=1.0,
                alpha=0.25,
            )  #

    #
    """
    font_size = 4
    # plot the MPO imbalance loss and cv loss in mpo_loss vertically
    plt.axhline(y=mpo_loss["imbalance_loss"], color="red", linestyle="--")
    plt.text(
        0,
        mpo_loss["imbalance_loss"],
        f"MPO imbalance loss: {mpo_loss['imbalance_loss']}",
        color="red",
        fontsize=font_size,
    )
    plt.axhline(y=mpo_loss["cv_loss"], color="red", linestyle="-.")
    plt.text(
        0,
        mpo_loss["cv_loss"],
        f"MPO CV loss: {mpo_loss['cv_loss']}",
        color="red",
        fontsize=font_size,
    )

    # plot the scFEA imbalance loss and cv loss in scfea_loss vertically
    plt.axhline(y=scfea_loss["imbalance_loss"], color="red", linestyle=":")
    plt.text(
        0,
        scfea_loss["imbalance_loss"],
        f"scFEA imbalance loss: {scfea_loss['imbalance_loss']}",
        color="red",
        fontsize=font_size,
    )
    plt.axhline(y=scfea_loss["cv_loss"], color="black", linestyle="-")
    plt.text(
        0,
        scfea_loss["cv_loss"],
        f"scFEA CV loss: {scfea_loss['cv_loss']}",
        color="red",
        fontsize=font_size,
    )
    """
    #
    plt.xticks(ticks=sorted(epochs))

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training, Validation, and Test Losses")
    plt.legend(loc="upper right", fontsize="small", ncol=2)
    plt.grid(True)
    plt.tight_layout()

    # save the plot
    plt.savefig(os.path.join(args.output_dir_path, "loss_curves.png"))


def get_one_sample_imbalanceLoss(vector, compounds_reactions_df):
    tmp1 = vector * compounds_reactions_df.values
    tmp2 = np.sum(tmp1, axis=1)
    tmp3 = abs(tmp2)
    tmp4 = np.sum(tmp3)
    tmp5 = np.round(tmp4, 3)
    return tmp5


def get_imbalanceLoss(compounds_reactions_df, samples_reactions_flux_df):
    samples_reactions_flux_df = (
        samples_reactions_flux_df.div(
            np.linalg.norm(samples_reactions_flux_df, axis=1), axis=0
        )
        * 1.0
    )
    imbalanceLoss_values = []
    n_processes = min(os.cpu_count(), samples_reactions_flux_df.shape[0])
    pool = Pool(n_processes)
    for row_i in samples_reactions_flux_df.values:
        imbalanceLoss_values.append(
            pool.apply_async(
                get_one_sample_imbalanceLoss, args=(row_i, compounds_reactions_df)
            )
        )
    pool.close()
    pool.join()
    imbalanceLoss_values = [res.get() for res in imbalanceLoss_values]

    return np.mean(imbalanceLoss_values)
