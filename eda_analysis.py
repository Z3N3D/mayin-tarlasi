"""
Exploratory Data Analysis for Aerial Landmine Detection Dataset
Generates: correlation_heatmap.png, outliers_distributions.png, feature_distributions.png
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE = Path("/home/hamzah/Desktop/beykoz/proje/MAYIN TARLASI")
CSV  = BASE / "landmine_tabular_data.csv"

FEATURES = ["area", "circularity", "mean_intensity", "thermal_contrast", "edge_density"]
LABEL    = "label"

PALETTE  = {0: "#4C9BE8", 1: "#E8564C"}          # blue = bg, red = mine
CLASS_NAMES = {0: "Background (0)", 1: "Mine (1)"}

sns.set_theme(style="whitegrid", font_scale=1.15)

# ── Load & clean ───────────────────────────────────────────────────────────
print("Loading CSV…")
df = pd.read_csv(CSV)
print(f"  Shape before cleaning: {df.shape}")

# Replace infinities with NaN, then drop
df.replace([np.inf, -np.inf], np.nan, inplace=True)
n_missing = df[FEATURES + [LABEL]].isna().sum().sum()
print(f"  Missing / inf values found: {n_missing}")
df.dropna(subset=FEATURES + [LABEL], inplace=True)
print(f"  Shape after cleaning: {df.shape}")

df[LABEL] = df[LABEL].astype(int)
df["Class"] = df[LABEL].map({0: "Background", 1: "Mine"})

print("\nClass distribution:")
print(df[LABEL].value_counts())

# ── 1. Correlation Heatmap ─────────────────────────────────────────────────
print("\nGenerating correlation heatmap…")
corr_cols = FEATURES + [LABEL]
corr = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(8, 6))
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask, k=1)] = False   # show full matrix

cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(
    corr, annot=True, fmt=".2f", cmap=cmap,
    vmin=-1, vmax=1, linewidths=0.5, ax=ax,
    annot_kws={"size": 11}
)
ax.set_title("Feature Correlation Heatmap\n(including target Label)", fontsize=14, fontweight="bold", pad=14)
plt.tight_layout()
out = BASE / "correlation_heatmap.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved → {out}")

# Print top correlations with label
print("\nTop correlations with Label:")
label_corr = corr[LABEL].drop(LABEL).sort_values(key=abs, ascending=False)
print(label_corr.to_string())

# ── 2. Boxplots per feature grouped by class ──────────────────────────────
print("\nGenerating boxplots…")
fig, axes = plt.subplots(1, len(FEATURES), figsize=(18, 6))
fig.suptitle("Feature Distributions by Class (Mine vs Background)\nBoxplots with Outliers",
             fontsize=14, fontweight="bold", y=1.01)

feature_labels = {
    "area":             "Area (px²)",
    "circularity":      "Circularity",
    "mean_intensity":   "Mean Thermal Intensity",
    "thermal_contrast": "Thermal Contrast",
    "edge_density":     "Edge Density",
}

for ax, feat in zip(axes, FEATURES):
    sns.boxplot(
        data=df, x="Class", y=feat,
        palette={"Background": PALETTE[0], "Mine": PALETTE[1]},
        order=["Background", "Mine"],
        width=0.5, fliersize=3, ax=ax
    )
    ax.set_title(feature_labels[feat], fontsize=12, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel(feature_labels[feat], fontsize=10)
    # Annotate medians
    for i, cls in enumerate(["Background", "Mine"]):
        med = df.loc[df["Class"] == cls, feat].median()
        ax.text(i, med, f"{med:.2f}", ha="center", va="bottom",
                fontsize=9, color="black", fontweight="bold")

plt.tight_layout()
out = BASE / "outliers_distributions.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved → {out}")

# ── 3. Histograms per feature ──────────────────────────────────────────────
print("\nGenerating histograms…")
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Feature Distributions by Class — Histogram Overlays",
             fontsize=14, fontweight="bold", y=1.01)
axes_flat = axes.flatten()

for ax, feat in zip(axes_flat, FEATURES):
    for label_val, color in PALETTE.items():
        subset = df.loc[df[LABEL] == label_val, feat]
        ax.hist(subset, bins=40, alpha=0.55, color=color,
                label=CLASS_NAMES[label_val], density=True, edgecolor="none")
        # KDE overlay
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(subset.dropna())
        xs = np.linspace(subset.min(), subset.max(), 300)
        ax.plot(xs, kde(xs), color=color, linewidth=2)
    ax.set_title(feature_labels[feat], fontsize=12, fontweight="bold")
    ax.set_xlabel(feature_labels[feat], fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.legend(fontsize=9)

# Hide the empty 6th subplot
axes_flat[-1].set_visible(False)

plt.tight_layout()
out = BASE / "feature_distributions.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved → {out}")

# ── Summary Stats ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("DATASET SUMMARY")
print("="*60)
print(f"Total samples  : {len(df):,}")
print(f"Features       : {FEATURES}")
print(f"Class balance  : {df[LABEL].value_counts().to_dict()}")
print("\nPer-class mean values:")
print(df.groupby("Class")[FEATURES].mean().round(3).to_string())
print("\nCorrelations with Label (sorted by |r|):")
print(label_corr.to_string())
print("="*60)
print("\nAll EDA plots saved successfully.")
