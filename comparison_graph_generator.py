import matplotlib.pyplot as plt
import pandas as pd

# Data for the models
models = ['YOLOv1', 'YOLOv3', 'SSD', 'Faster R-CNN']
mAP = [90.93, 78.6, 76.3, 79.8]
inference_time = [22, 22, 28, 120]

# Creating the DataFrame
df = pd.DataFrame({
    'Model': models,
    'mAP': mAP,
    'Inference Time (ms)': inference_time
})

# Plotting the comparison graphs
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot mAP
color = 'tab:blue'
ax1.set_xlabel('Model')
ax1.set_ylabel('mAP (%)', color=color)
ax1.bar(df['Model'], df['mAP'], color=color, alpha=0.6)
ax1.tick_params(axis='y', labelcolor=color)

# Creating a twin axis to plot inference time
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Inference Time (ms)', color=color)
ax2.plot(df['Model'], df['Inference Time (ms)'], color=color, marker='o')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('Model Performance Comparison')

# Save the figure as SVG
plt.savefig('Models_Performance_Comparison.svg', format='svg')

plt.show()
