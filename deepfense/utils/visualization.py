import matplotlib.pyplot as plt
import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
except ImportError:
    sns = None

def plot_metric_trend(history_dict, metric_name, save_path, title=None, xlabel="Epoch"):
    """
    Plots the trend of a metric over epochs or steps. 
    Supports comparing multiple series (e.g. Train vs Val).
    
    Args:
        history_dict: Dict { "Train": [(x, val)...], "Val": [(x, val)...] } 
                      OR List of tuples [(x, val)...] for single series.
        metric_name: Name of the metric (y-axis label).
        save_path: File path to save the plot.
        title: Plot title.
        xlabel: Label for x-axis ("Epoch" or "Step").
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Normalize input to dict format
    if isinstance(history_dict, list):
        history_dict = {"Series": history_dict}
        
    all_data = []
    for series_name, history in history_dict.items():
        if not history:
            continue
        x_vals, values = zip(*history)
        # Clean tensor values
        values = [v.item() if hasattr(v, 'item') else v for v in values]
        
        for x, v in zip(x_vals, values):
            all_data.append({xlabel: x, metric_name: v, "Split": series_name})
            
    if not all_data:
        plt.close(fig)
        return

    df = pd.DataFrame(all_data)
    
    if sns:
        # Use seaborn for nice multi-line plots
        sns.lineplot(data=df, x=xlabel, y=metric_name, hue="Split", marker="o", ax=ax, linewidth=2)
    else:
        # Fallback matplotlib
        for series_name, history in history_dict.items():
            x_vals, values = zip(*history)
            values = [v.item() if hasattr(v, 'item') else v for v in values]
            ax.plot(x_vals, values, marker='o', linewidth=2, label=series_name)
        ax.legend()
        ax.grid(True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(metric_name)
        
    ax.set_title(title or f"{metric_name} Trend")
    
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
