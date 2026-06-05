import matplotlib.pyplot as plt

# knn update
x_labels = ["fixed", "add", "update", "delayed"]

mcar = [15.9, 14.9, 14.3, 14.6]
mar  = [16.7, 15.8, 14.9, 16.2]
mnar = [18.1, 17.9, 17.4, 17.8]

plt.figure(figsize=(8, 5))

plt.plot(x_labels, mcar, marker="o", linewidth=2, label="MCAR")
plt.plot(x_labels, mar, marker="s", linewidth=2, label="MAR")
plt.plot(x_labels, mnar, marker="^", linewidth=2, label="MNAR")

plt.xlabel("Impact of knn graph update type")
plt.ylabel("MAPE")
# plt.title("MAPE across Model Settings")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

plt.savefig("figures/knn-graph-ablation.svg", format="svg", bbox_inches="tight")
plt.show()


#

# Fusion

x_labels = ["mean", "learnable", "reliability-aware"]

mcar = [17.2, 14.1, 14.3]
mar  = [19.0, 15.2, 14.9]
mnar = [19.2, 17.9, 17.4]

plt.figure(figsize=(8, 5))

plt.plot(x_labels, mcar, marker="o", linewidth=2, label="MCAR")
plt.plot(x_labels, mar, marker="s", linewidth=2, label="MAR")
plt.plot(x_labels, mnar, marker="^", linewidth=2, label="MNAR")

plt.xlabel("Impact of fusion type")
plt.ylabel("MAPE")
# plt.title("MAPE across Model Settings")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

plt.savefig("figures/fusion-ablation.svg", format="svg", bbox_inches="tight")
plt.show()




# e2e

x_labels = ["end-2-end", "pretrain"]

mcar = [14.6, 14.3]
mar  = [15.0, 14.9]
mnar = [17.8, 17.4]

plt.figure(figsize=(8, 5))

plt.plot(x_labels, mcar, marker="o", linewidth=2, label="MCAR")
plt.plot(x_labels, mar, marker="s", linewidth=2, label="MAR")
plt.plot(x_labels, mnar, marker="^", linewidth=2, label="MNAR")

plt.xlabel("Impact of end-to-end training vs. pretraining")
plt.ylabel("MAPE")
# plt.title("MAPE across Model Settings")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

plt.savefig("figures/e2e-ablation.svg", format="svg", bbox_inches="tight")
plt.show()



# recon loss

x_labels = ["without recon loss", "with recon loss"]

mcar = [14.5, 14.3]
mar  = [15.7, 14.9]
mnar = [18.9, 17.4]

plt.figure(figsize=(8, 5))

plt.plot(x_labels, mcar, marker="o", linewidth=2, label="MCAR")
plt.plot(x_labels, mar, marker="s", linewidth=2, label="MAR")
plt.plot(x_labels, mnar, marker="^", linewidth=2, label="MNAR")

plt.xlabel("Impact of having reconstruction objective")
plt.ylabel("MAPE")
# plt.title("MAPE across Model Settings")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

plt.savefig("figures/recon-ablation.svg", format="svg", bbox_inches="tight")
plt.show()