import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# CAS-V Ablation Study Results
# ============================================================

methods = [
    "Baseline\n(No CAS-V)",
    "+ Visual\nSimilarity",
    "+ Conflict\nDetection",
    "Full CAS-V"
]

# Experimental results
clap_similarity = [62, 68, 73, 78]
conflict_rate = [24, 18, 12, 8]

x = np.arange(len(methods))

# ============================================================
# Figure 1: CLAP Similarity
# ============================================================

plt.figure(figsize=(8, 5))

bars = plt.bar(x, clap_similarity)

plt.xticks(x, methods, fontsize=11)
plt.ylabel("CLAP Similarity (%)", fontsize=12)
plt.xlabel("Method", fontsize=12)
plt.title("Impact of CAS-V Components on CLAP Similarity", fontsize=14)

plt.ylim(50, 85)
plt.grid(axis="y", linestyle="--", alpha=0.5)

# Add values on top of bars
for bar, value in zip(bars, clap_similarity):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.8,
        f"{value}%",
        ha="center",
        fontsize=11,
        fontweight="bold"
    )

plt.tight_layout()
plt.show()


# ============================================================
# Figure 2: Conflict Rate
# ============================================================

plt.figure(figsize=(8, 5))

bars = plt.bar(x, conflict_rate)

plt.xticks(x, methods, fontsize=11)
plt.ylabel("Conflict Rate (%)", fontsize=12)
plt.xlabel("Method", fontsize=12)
plt.title("Impact of CAS-V Components on Conflict Rate", fontsize=14)

plt.ylim(0, 30)
plt.grid(axis="y", linestyle="--", alpha=0.5)

# Add values on top of bars
for bar, value in zip(bars, conflict_rate):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.8,
        f"{value}%",
        ha="center",
        fontsize=11,
        fontweight="bold"
    )

plt.tight_layout()
plt.show()