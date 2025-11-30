import matplotlib.pyplot as plt
import numpy as np
import logging
import pandas as pd

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
except ImportError:
    sns = None
    logger.warning("Seaborn not found. Plots will look less polished.")

try:
    from sklearn.manifold import TSNE
    from sklearn.metrics import det_curve
except ImportError:
    TSNE = None
    det_curve = None
    logger.warning("scikit-learn not found. Visualization functions requiring it will fail.")


def plot_metric_trend(history, metric_name, title=None):
    """
    Plots the trend of a metric over epochs.
    history: List of tuples (epoch, value)
    """
    if not history:
        return None
        
    epochs, values = zip(*history)
    # Clean values (handle tensors)
    values = [v.item() if hasattr(v, 'item') else v for v in values]
    
    data = pd.DataFrame({"Epoch": epochs, metric_name: values})
    
    fig, ax = plt.subplots(figsize=(10, 6))
    if sns:
        sns.lineplot(data=data, x="Epoch", y=metric_name, marker="o", ax=ax, linewidth=2)
    else:
        ax.plot(epochs, values, marker='o', linewidth=2)
        ax.grid(True)
        
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_name)
    ax.set_title(title or f"{metric_name} Trend")
    
    plt.close(fig)
    return fig


def plot_det_curve(labels, scores, title="DET Curve"):
    """
    Plots Detection Error Trade-off (DET) curve on log-log scale.
    """
    if det_curve is None:
        return None

    fpr, fnr, thresholds = det_curve(labels, scores, pos_label=1)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot using standard matplotlib for robust log-scale handling
    ax.plot(fpr, fnr, label="System", linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label="EER Reference")
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('False Negative Rate (FNR)')
    ax.set_title(title)
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend()
    
    # Set standard ASV limits for visibility
    ax.set_xlim([0.0001, 1.0])
    ax.set_ylim([0.0001, 1.0])
    
    plt.close(fig)
    return fig


def plot_score_hist(labels, scores, title="Score Distribution"):
    """
    Plots histogram of scores with KDE.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare DataFrame
    # Handle tensor inputs if necessary
    if hasattr(labels, "cpu"): labels = labels.cpu().numpy()
    if hasattr(scores, "cpu"): scores = scores.cpu().numpy()
    
    data = pd.DataFrame({
        "Score": scores,
        "Label": ["Bonafide" if l==1 else "Spoof" for l in labels]
    })
    
    if sns:
        sns.histplot(
            data=data, x="Score", hue="Label", 
            kde=True, element="step", ax=ax, 
            palette={"Bonafide": "blue", "Spoof": "red"},
            alpha=0.5
        )
    else:
        # Fallback Matplotlib
        bonafide = scores[labels==1]
        spoof = scores[labels==0]
        ax.hist(bonafide, bins=50, alpha=0.5, color='blue', label='Bonafide', density=True)
        ax.hist(spoof, bins=50, alpha=0.5, color='red', label='Spoof', density=True)
        ax.legend()
        
    ax.set_title(title)
    plt.close(fig)
    return fig


def plot_tsne(embeddings, labels, max_samples=2000, title="t-SNE"):
    """
    Plots t-SNE of embeddings.
    """
    if TSNE is None: return None
    if len(labels) == 0: return None
    
    # Handle tensor inputs
    if hasattr(labels, "cpu"): labels = labels.cpu().numpy()
    if hasattr(embeddings, "cpu"): embeddings = embeddings.cpu().numpy()

    # Downsample
    if len(labels) > max_samples:
        idx = np.random.choice(len(labels), max_samples, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]
        
    try:
        tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
        emb_2d = tsne.fit_transform(embeddings)
    except Exception as e:
        logger.warning(f"t-SNE failed: {e}")
        return None
        
    data = pd.DataFrame({
        "x": emb_2d[:, 0],
        "y": emb_2d[:, 1],
        "Label": ["Bonafide" if l==1 else "Spoof" for l in labels]
    })
    
    fig, ax = plt.subplots(figsize=(10, 10))
    if sns:
        sns.scatterplot(
            data=data, x="x", y="y", hue="Label", 
            alpha=0.6, s=40, ax=ax, 
            palette={"Bonafide": "blue", "Spoof": "red"}
        )
    else:
        idx_bona = labels == 1
        idx_spoof = labels == 0
        ax.scatter(emb_2d[idx_bona, 0], emb_2d[idx_bona, 1], c='blue', label='Bonafide', alpha=0.5, s=15)
        ax.scatter(emb_2d[idx_spoof, 0], emb_2d[idx_spoof, 1], c='red', label='Spoof', alpha=0.5, s=15)
        ax.legend()
        
    ax.set_title(title)
    plt.close(fig)
    return fig
