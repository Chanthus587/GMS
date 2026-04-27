import pandas as pd
import matplotlib.pyplot as plt
import os

# ==============================
# CONFIG
# ==============================
CLEAN_PATH = "outputs/gms_clean.csv"
NOISE_PATH = "outputs/gms_noise.csv"
OUT_DIR = "outputs/plots"

os.makedirs(OUT_DIR, exist_ok=True)


# ==============================
# LOAD + ALIGN DATA
# ==============================
def load_and_align():
    clean = pd.read_csv(CLEAN_PATH)
    noise = pd.read_csv(NOISE_PATH)

    # align by common time range
    max_time = min(clean['time'].max(), noise['time'].max())

    clean = clean[clean['time'] <= max_time]
    noise = noise[noise['time'] <= max_time]

    return clean, noise


# ==============================
# METRICS
# ==============================
def compute_metrics(df):
    TP = ((df.pred==1)&(df.truth==1)).sum()
    FP = ((df.pred==1)&(df.truth==0)).sum()
    TN = ((df.pred==0)&(df.truth==0)).sum()
    FN = ((df.pred==0)&(df.truth==1)).sum()

    acc = (TP+TN)/(TP+TN+FP+FN)
    precision = TP/(TP+FP) if TP+FP else 0
    recall = TP/(TP+FN) if TP+FN else 0
    far = FP/(FP+TN) if FP+TN else 0
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0

    return [acc, precision, recall, far, f1]


# ==============================
# METRIC COMPARISON (GOOD VERSION)
# ==============================
def plot_metrics(clean, noise):
    labels = ["Accuracy", "Precision", "Recall", "FAR", "F1"]

    m_clean = compute_metrics(clean)
    m_noise = compute_metrics(noise)

    x = range(len(labels))
    width = 0.35

    plt.figure()
    plt.bar([i - width/2 for i in x], m_clean, width)
    plt.bar([i + width/2 for i in x], m_noise, width)

    plt.xticks(x, labels)
    plt.title("Performance Comparison (Clean vs Noise)")
    plt.savefig(f"{OUT_DIR}/metrics_comparison.png")
    plt.close()


# ==============================
# GMS vs TIME
# ==============================
def plot_gms_time(df, node, name):
    nd = df[df["node"] == node]

    plt.figure()
    plt.plot(nd["time"], nd["gms"])
    plt.title(f"GMS over Time (Node {node}) - {name}")
    plt.xlabel("Time")
    plt.ylabel("GMS Score")
    plt.savefig(f"{OUT_DIR}/gms_time_node{node}_{name}.png")
    plt.close()


# ==============================
# TEMP vs GMS
# ==============================
def plot_temp_vs_gms(df, node, name):
    nd = df[df["node"] == node]

    plt.figure()
    plt.plot(nd["time"], nd["temp"])
    plt.plot(nd["time"], nd["gms"])
    plt.title(f"Temp vs GMS (Node {node}) - {name}")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.savefig(f"{OUT_DIR}/temp_vs_gms_node{node}_{name}.png")
    plt.close()


# ==============================
# DISTRIBUTION
# ==============================
def plot_distribution(clean, noise):
    plt.figure()
    plt.hist(clean["gms"], bins=50, alpha=0.5)
    plt.hist(noise["gms"], bins=50, alpha=0.5)
    plt.title("GMS Distribution (Clean vs Noise)")
    plt.savefig(f"{OUT_DIR}/gms_distribution.png")
    plt.close()


# ==============================
# HEATMAP
# ==============================
def plot_heatmap(df, name):
    # FIX: handle duplicates safely
    pivot = df.pivot_table(
        index="node",
        columns="time",
        values="gms",
        aggfunc="mean"   # average duplicates
    )

    plt.figure()
    plt.imshow(pivot, aspect='auto')
    plt.title(f"GMS Heatmap - {name}")
    plt.xlabel("Time")
    plt.ylabel("Node")
    plt.colorbar()
    plt.savefig(f"{OUT_DIR}/heatmap_{name}.png")
    plt.close()

# ==============================
# MAIN
# ==============================
def main():
    clean, noise = load_and_align()

    # Metrics
    plot_metrics(clean, noise)

    # Node plots
    plot_gms_time(clean, 0, "clean")
    plot_gms_time(noise, 0, "noise")

    plot_temp_vs_gms(clean, 0, "clean")
    plot_temp_vs_gms(noise, 0, "noise")

    # Distribution
    plot_distribution(clean, noise)

    # Heatmaps
    plot_heatmap(clean, "clean")
    plot_heatmap(noise, "noise")

    print("Plots saved in:", OUT_DIR)


if __name__ == "__main__":
    main()