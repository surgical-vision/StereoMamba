import matplotlib.pyplot as plt
# Increase default font sizes
plt.rcParams.update({'font.size': 12})  # Increase base font size
# data
# x = [2.59, 2.58, 2.58, 2.56, 2.65, 2.84, 2.60, 2.55]
# y = [3.22, 2.7, 5.00, 2.44, 1.96, 100.00, 1.08, 21.28]
# x = [0.9120, 0.9141, 0.9108, 0.9096, 0.9110, 0.8842, 0.9134, 0.9149]
# y = [17.1222, 17.3140, 17.1873, 16.9007, 16.8081, 15.6194, 17.2028, 17.3054]
# x = [0.8697, 0.8696, 0.8637, 0.8641, 0.8793, 0.8526, 0.8755, 0.8790]
# y = [14.5847, 14.4944, 14.5558, 14.3403, 14.7840, 14.1028, 14.6991, 14.8468]
x = [0.8908, 0.8919, 0.8873, 0.8869, 0.8952, 0.8684, 0.8945, 0.8970]
y = [15.8534, 15.9042, 15.8715, 15.6205, 15.7961, 14.8611, 15.9509, 16.0761]
labels = ["GwcNet-gc", "ACVNet", "RAFT-Stereo", "IGEV-Stereo", 
          "Selective-Stereo", "MSDESIS", "Shi et al.", "StereoMamba(ours)"]
markers = ['o', '^', 's', 'p', 'h', 'o', 'd', '*']
colors = ['blue', 'cyan', 'green', 'black', 'magenta', 'orange', 'purple', 'red']

# plot
plt.figure(figsize=(6,6))
for i in range(len(x)):
    if labels[i] == "StereoMamba(ours)":
        plt.scatter(x[i], y[i], color=colors[i], marker=markers[i], s=200, label=r"$\bf{" + labels[i].replace("(", "{(}") + "}$")
    else:
        plt.scatter(x[i], y[i], color=colors[i], marker=markers[i], s=100, label=labels[i])

# axis label
plt.xlabel("SSIM", fontsize=14)
plt.ylabel("PSNR", fontsize=14)
plt.legend(loc='upper left', fontsize=12)

# Increase tick label sizes
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# save image
plt.savefig("generalization.png", dpi=300, bbox_inches='tight')
