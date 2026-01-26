import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score

def plot_confusion_matrix(
    y_true,
    y_pred,
    title="Confusion Matrix",
    title_prefix="",
    save_path=None
):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)

    # prepare save directory if provided
    save_dir = None
    if save_path is not None:
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', cbar=False,
        xticklabels=['Legit', 'Fraud'],
        yticklabels=['Legit', 'Fraud']
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.tight_layout()

    # save
    if save_dir is not None:
        filename = f"{title_prefix}confusion_matrix.png" if title_prefix else "confusion_matrix.png"
        plt.savefig(save_dir / filename, dpi=300, bbox_inches="tight")

    plt.show()


def _add_annotation_legend_with_arrows(ax, points, title=None,
                                      x_text=1.02, y_top=0.90, y_step=0.07):
    if title is not None:
        ax.text(
            x_text, min(0.98, y_top + 0.07), title,
            transform=ax.transAxes, ha="left", va="center",
            fontsize=10, fontweight="bold", clip_on=False
        )

    for i, (lab, x_pt, y_pt) in enumerate(points):
        y_txt = y_top - i * y_step
        ax.annotate(
            lab,
            xy=(x_pt, y_pt), xycoords="data",
            xytext=(x_text, y_txt), textcoords="axes fraction",
            ha="left", va="center",
            arrowprops=dict(arrowstyle="->", lw=1.0),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.5),
            annotation_clip=False
        )


def plot_roc_pr_with_conformal_points(
    y_true,
    scores,
    taus_by_alpha,
    best_f1_tau=None,
    f1=None,
    title_prefix="",
    save_path=None
):
    """
    y_true: 0/1 labels
    scores: higher => more fraud
    taus_by_alpha: dict {alpha: tau}
    best_f1_tau: scalar threshold corresponding to best F1 score (optional)
    """
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    scores = np.asarray(scores).reshape(-1)

    # prepare save directory if needed
    save_dir = None
    if save_path is not None:
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

    # ============================
    # ROC
    # ============================
    fpr_curve, tpr_curve, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr_curve, tpr_curve)

    plt.figure(figsize=(9, 6))
    ax = plt.gca()
    ax.plot(fpr_curve, tpr_curve, label=f"ROC curve (AUC={roc_auc:.4f})")

    roc_points = []

    # conformal taus -> red points
    for a, tau in taus_by_alpha.items():
        y_pred = (scores > tau).astype(int)
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        fpr = fp / (fp + tn + 1e-12)
        tpr = tp / (tp + fn + 1e-12)

        ax.scatter([fpr], [tpr], c="red", s=60, zorder=5)
        roc_points.append((f"α={a:.2%}", fpr, tpr))

    # best F1 threshold -> green point
    if best_f1_tau is not None:
        y_pred_best = (scores > best_f1_tau).astype(int)
        fp = ((y_true == 0) & (y_pred_best == 1)).sum()
        tn = ((y_true == 0) & (y_pred_best == 0)).sum()
        tp = ((y_true == 1) & (y_pred_best == 1)).sum()
        fn = ((y_true == 1) & (y_pred_best == 0)).sum()
        fpr_best = fp / (fp + tn + 1e-12)
        tpr_best = tp / (tp + fn + 1e-12)

        ax.scatter(
            [fpr_best],
            [tpr_best],
            c="green",
            s=70,
            zorder=6,
            label=f"best f1 score treshold: f1={f1}",
        )

    roc_points.sort(key=lambda t: t[1])
    _add_annotation_legend_with_arrows(ax, roc_points, title="Conformal points")

    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title(f"{title_prefix}ROC with conformal operating points")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")
    plt.tight_layout(rect=[0, 0, 0.82, 1])

    # save ROC if requested
    if save_dir is not None:
        roc_filename = f"{title_prefix}roc.png" if title_prefix else "roc.png"
        plt.savefig(save_dir / roc_filename, dpi=300, bbox_inches="tight")

    plt.show()

    # ============================
    # PR
    # ============================
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)

    plt.figure(figsize=(9, 6))
    ax = plt.gca()
    ax.plot(recall_curve, precision_curve, label=f"PR curve (AP={ap:.4f})")

    pr_points = []

    # conformal taus -> red points
    for a, tau in taus_by_alpha.items():
        y_pred = (scores > tau).astype(int)
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()

        recall = tp / (tp + fn + 1e-12)
        precision = tp / (tp + fp + 1e-12)

        ax.scatter([recall], [precision], c="red", s=60, zorder=5)
        pr_points.append((f"α={a:.2%}", recall, precision))

    # best F1 threshold -> green point
    if best_f1_tau is not None:
        y_pred_best = (scores > best_f1_tau).astype(int)
        fp = ((y_true == 0) & (y_pred_best == 1)).sum()
        tp = ((y_true == 1) & (y_pred_best == 1)).sum()
        fn = ((y_true == 1) & (y_pred_best == 0)).sum()

        recall_best = tp / (tp + fn + 1e-12)
        precision_best = tp / (tp + fp + 1e-12)

        ax.scatter(
            [recall_best],
            [precision_best],
            c="green",
            s=70,
            zorder=6,
            label=f"best f1 score treshold: f1={f1}",
        )

    pr_points.sort(key=lambda t: t[1])
    _add_annotation_legend_with_arrows(ax, pr_points, title="Conformal points")

    ax.set_xlabel("Recall (TPR)")
    ax.set_ylabel("Precision")
    ax.set_title(f"{title_prefix}PR with conformal operating points")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left")
    plt.tight_layout(rect=[0, 0, 0.82, 1])

    # save PR if requested
    if save_dir is not None:
        pr_filename = f"{title_prefix}pr.png" if title_prefix else "pr.png"
        plt.savefig(save_dir / pr_filename, dpi=300, bbox_inches="tight")

    plt.show()