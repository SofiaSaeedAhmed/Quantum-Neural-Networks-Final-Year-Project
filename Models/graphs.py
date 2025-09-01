import matplotlib.pyplot as plt
import numpy as np

# graphs for 2 bars

# Data
models = ["0.001", "0.0025/0.001",
          "0.005/0.001", "0.005/0.003"]


val_acc_dropout = [0.8274, 0.8212, 0.8370, 0.8381]
val_acc_nodropout = [0.5199, 0.5325, 0.4946, 0.4886]

# Reverse for top-down order
models = models[::-1]
val_acc_dropout = val_acc_dropout[::-1]
val_acc_nodropout = val_acc_nodropout[::-1]
y_pos = np.arange(len(models))

# Plot
fig, ax = plt.subplots(figsize=(12, 6))

# Thinner bars (height reduced from 0.3 to 0.2)
bar1 = ax.barh(y_pos - 0.1, val_acc_dropout, height=0.2, label='_nolegend_', color='#001f3f')
bar2 = ax.barh(y_pos + 0.1, val_acc_nodropout, height=0.2, label='_nolegend_', color='#66b2ff')

# Add values to bars
for bars in [bar1, bar2]:
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.002, bar.get_y() + bar.get_height()/2,
                f'{width:.4f}', va='center', ha='left', fontsize=10)

# Axis setup
ax.set_yticks(y_pos)
ax.set_yticklabels(models, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel("Validation Accuracy and Loss on C32C64C128C256FC128FC20Q4", fontsize=11)
ax.set_title("Best Results After 30 Experiments", fontsize=13, pad=30)

# Add custom dot markers as legend
legend_labels = ['Validation Accuracy', 'Validation Loss']
legend_colors = ["#001f3f", "#66b2ff"]
for i, (label, color) in enumerate(zip(legend_labels, legend_colors)):
    ax.plot([], [], 'o', color=color, label=label)

# Place legend below title with gap
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=2, fontsize=10, frameon=False)

# Remove unnecessary spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)

# Adjust layout
plt.tight_layout(pad=2)
plt.show()


# graphs for 3 bars
#
# import matplotlib.pyplot as plt
# import numpy as np
#
# # Data
# models = ["E0", "E2",
#           "E3", "E5", "E10", "E2S"]
#
# val_loss_adam = [0.7877, 0.8132, 0.8274, 0.8186, 0.8179, 0.8041]
# val_loss_sgd = [0.6369, 0.5667, 0.5199, 0.5451, 0.5391, 0.5808]
# val_loss_rmsprop = [4543.62, 6799.56, 8962.33, 7245.36, 8284.28, 7237.63]
#
#
# # Reverse for top-down order
# models = models[::-1]
# val_loss_adam = val_loss_adam[::-1]
# val_loss_sgd = val_loss_sgd[::-1]
# val_loss_rmsprop = val_loss_rmsprop[::-1]
# y_pos = np.arange(len(models))
#
# # Plot
# fig, ax = plt.subplots(figsize=(12, 6))
#
# bar_spacing = 0.2
# bar1 = ax.barh(y_pos - bar_spacing, val_loss_adam, height=0.2, label='_nolegend_', color='#001f3f')
# bar2 = ax.barh(y_pos, val_loss_sgd, height=0.2, label='_nolegend_', color='#66b2ff')
# bar3 = ax.barh(y_pos + bar_spacing, val_loss_rmsprop, height=0.2, label='_nolegend_', color='#2ca02c')  # green
#
# # Add values to bars
# for bars in [bar1, bar2, bar3]:
#     for bar in bars:
#         width = bar.get_width()
#         ax.text(width + 0.002, bar.get_y() + bar.get_height()/2,
#                 f'{width:.4f}', va='center', ha='left', fontsize=10)
#
# # Axis setup
# ax.set_yticks(y_pos)
# ax.set_yticklabels(models, fontsize=10)
# ax.invert_yaxis()
# ax.set_xlabel("Validation Loss", fontsize=11)
# ax.set_title("Best Results After 30 Experiments", fontsize=13, pad=30)
#
# # Custom legend using dots
# legend_labels = ['Adam / No dropout', 'Adam / Dropout', 'SGD / Dropout']
# legend_colors = ["#001f3f", "#66b2ff", "#2ca02c"]
# for label, color in zip(legend_labels, legend_colors):
#     ax.plot([], [], 'o', color=color, label=label)
#
# # Legend placement
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=3, fontsize=10, frameon=False)
#
# # Remove unnecessary spines
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['left'].set_visible(False)
#
# # Layout tweak
# plt.tight_layout(pad=2.5)
# plt.show()
