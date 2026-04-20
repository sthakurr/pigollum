import os
import gc
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from umap import UMAP

from gollum.featurization.text import get_huggingface_embeddings, get_tokens
from gollum.featurization.deep import LLMFeaturizer


def load_split_labels(data_path):
    """Load the full dataset and derive 3-way split labels.

    Expects columns: sequence, target, set (train/test), validation (bool).
    Returns:
        df: Full DataFrame
        labels: np.ndarray of ints (0=train, 1=validation, 2=test)
        label_names: dict mapping int -> str
    """
    df = pd.read_csv(data_path)
    labels = np.full(len(df), -1, dtype=int)
    labels[(df["set"] == "train") & (~df["validation"])] = 0
    labels[(df["set"] == "train") & (df["validation"])] = 1
    labels[df["set"] == "test"] = 2
    label_names = {0: "train", 1: "validation", 2: "test"}
    return df, labels, label_names


def extract_original_embeddings(sequences, model_name, pooling_method, batch_size=8):
    """Extract embeddings from the original frozen ESM model."""
    embeddings = get_huggingface_embeddings(
        sequences,
        model_name=model_name,
        pooling_method=pooling_method,
        batch_size=batch_size,
    )
    gc.collect()
    torch.cuda.empty_cache()
    return embeddings


def extract_finetuned_embeddings(
    sequences, model_name, model_state_path, model_config, batch_size=4
):
    """Load the finetuned LLMFeaturizer and extract embeddings.

    Args:
        sequences: list of raw sequences
        model_name: HuggingFace model name (for tokenization)
        model_state_path: path to saved state dict
        model_config: dict with LLMFeaturizer init_args
        batch_size: batch size for embedding extraction
    Returns:
        np.ndarray of shape [n_sequences, embedding_dim]
    """
    tokenized = get_tokens(sequences, model_name=model_name)
    tokenized_x = torch.from_numpy(tokenized).to(torch.float64).cuda()

    model = LLMFeaturizer(**model_config)
    state_dict = torch.load(model_state_path, map_location="cuda", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    model = model.to(dtype=torch.float64, device="cuda")

    with torch.no_grad():
        embeddings = model.get_embeddings(tokenized_x, batch_size=batch_size)

    embeddings_np = embeddings.cpu().numpy()

    del model, tokenized_x, embeddings
    gc.collect()
    torch.cuda.empty_cache()
    return embeddings_np


def reduce_umap(embeddings, n_neighbors=15, min_dist=0.1, random_state=42):
    """Reduce embeddings to 2D via UMAP."""
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    return reducer.fit_transform(embeddings)


def _plot_single(
    embeddings_2d, target_values, title, save_path, figsize=(10, 8), cmap="viridis"
):
    """Create a single scatter plot colored by target value."""
    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=target_values,
        cmap=cmap,
        s=15,
        alpha=0.7,
        edgecolors="none",
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Target Value")
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def _plot_all_splits(
    embeddings_2d,
    target_values,
    labels,
    label_names,
    title,
    save_path,
    figsize=(10, 8),
    cmap="viridis",
):
    """Plot all splits on one figure, using marker shapes per split, colored by target."""
    markers = {0: "o", 1: "s", 2: "^"}
    fig, ax = plt.subplots(figsize=figsize)

    for split_id in sorted(label_names.keys()):
        mask = labels == split_id
        sc = ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=target_values[mask],
            cmap=cmap,
            s=20,
            alpha=0.7,
            marker=markers[split_id],
            label=label_names[split_id],
            edgecolors="none",
            vmin=target_values.min(),
            vmax=target_values.max(),
        )

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Target Value")
    ax.legend(loc="best", framealpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def generate_plots(embeddings_2d, target_values, labels, label_names, prefix, output_dir):
    """Generate all plots for a single embedding source (original or finetuned)."""
    os.makedirs(output_dir, exist_ok=True)

    # All splits combined
    _plot_all_splits(
        embeddings_2d,
        target_values,
        labels,
        label_names,
        title=f"{prefix} — All Splits",
        save_path=os.path.join(output_dir, f"{prefix}_all_splits.png"),
    )

    # Per-split plots
    for split_id, split_name in label_names.items():
        mask = labels == split_id
        _plot_single(
            embeddings_2d[mask],
            target_values[mask],
            title=f"{prefix} — {split_name}",
            save_path=os.path.join(output_dir, f"{prefix}_{split_name}.png"),
        )


def visualize_embeddings(
    full_data_path,
    model_name,
    pooling_method,
    model_state_path,
    finetuning_model_config,
    output_dir="plots/embeddings",
    random_state=42,
):
    """Main entry point: extract embeddings, reduce with UMAP, generate plots.

    Args:
        full_data_path: path to CSV with sequence, target, set, validation columns
        model_name: HuggingFace model name (e.g. facebook/esm2_t33_650M_UR50D)
        pooling_method: pooling strategy (e.g. average)
        model_state_path: path to saved finetuned LLMFeaturizer state dict
        finetuning_model_config: dict of LLMFeaturizer init_args
        output_dir: directory to save plots
        random_state: seed for UMAP
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load data and split labels
    df, labels, label_names = load_split_labels(full_data_path)
    sequences = df["sequence"].tolist()
    target_values = df["target"].values

    # --- Original ESM embeddings ---
    print("Extracting original ESM embeddings...")
    original_emb = extract_original_embeddings(sequences, model_name, pooling_method)

    print("Running UMAP on original embeddings...")
    original_2d = reduce_umap(original_emb, random_state=random_state)
    del original_emb

    generate_plots(
        original_2d, target_values, labels, label_names, "original_esm", output_dir
    )
    del original_2d

    # --- Finetuned ESM embeddings ---
    print("Extracting finetuned ESM embeddings...")
    finetuned_emb = extract_finetuned_embeddings(
        sequences, model_name, model_state_path, finetuning_model_config
    )

    print("Running UMAP on finetuned embeddings...")
    finetuned_2d = reduce_umap(finetuned_emb, random_state=random_state)
    del finetuned_emb

    generate_plots(
        finetuned_2d, target_values, labels, label_names, "finetuned_esm", output_dir
    )
    del finetuned_2d

    print(f"All plots saved to {output_dir}")
