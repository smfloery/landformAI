import matplotlib.pyplot as plt
import pandas as pd
import os

# run_dir = r"D:\1_PROJECTS\01_SEHAG\20_GMK\runs\gmki_20250418_140426"
# train_log_path = os.path.join(run_dir, "train.log")

def plot_train_val_loss(log_path):

    out_path = os.path.join(os.path.dirname(log_path), "train_val_loss.png")
    pd_train_log = pd.read_csv(log_path, sep=";")

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    ax1.plot(pd_train_log.epoch, pd_train_log.train_loss, label="Training")
    ax1.plot(pd_train_log.epoch, pd_train_log.valid_loss, label="Validation")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--")

    ax2.plot(pd_train_log.epoch, pd_train_log.valid_iou_micro, label="IOU")
    ax2.plot(pd_train_log.epoch, pd_train_log.valid_f1_micro, label="F1")
    ax2.set_title("Quality")
    ax2.legend()
    ax2.grid(True, linestyle="--")

    plt.savefig(out_path, dpi=300, bbox_inches="tight")