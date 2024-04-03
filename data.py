import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

# Generate 10 cities -- call them A, ..., J
cities = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
hubs = ['B', 'F', 'I']
colors = ['blue' if city in hubs else 'red' for city in cities]
markers = ['*' if city in hubs else 'o' for city in cities]
sizes = [350 if city in hubs else 150 for city in cities]
text_sizes = [15 if city in hubs else 10 for city in cities]
labels = ['hub' if city in hubs else 'town' for city in cities]

np.random.seed(12)
# Generate random x and y coordinates
X = np.random.randint(0, 100, len(cities))
Y = np.random.randint(0, 100, len(cities))

# Sort cities, X, and Y, based on x position
sorted_cities = sorted(cities, key=lambda city: dict(zip(cities, X))[city])
X_ordered = sorted(X)
Y_ordered = sorted(Y, key=lambda y: dict(zip(Y, X))[y])

# Set seed
sequence_matrix = np.zeros((len(sorted_cities), len(sorted_cities)))
for i, city in enumerate(sorted_cities):
    sequence = np.zeros((len(sorted_cities)))
    if city in hubs:
        for j, city_2 in enumerate(sorted_cities):
            if city == city_2:
                sequence[j] = 0
            elif city_2 in hubs:
                sequence[j] = np.random.randint(500, 1000)
            else:
                sequence[j] = np.random.randint(50, 250)
    else:
        for j, city_2 in enumerate(cities):
            if city_2 in hubs:
                sequence[j] = np.random.randint(50, 100)
            else:
                # sequence[j] = np.random.choice(np.arange(3), p=[0.50, 0.25, 0.25])
                sequence[j] = np.random.randint(10, 50)
        
    sequence_matrix[i] = sequence
for i in range(len(sorted_cities)):
    for j in range(len(sorted_cities)):
        if j <= i:
            sequence_matrix[i, j] = 0

# Getting the Upper Triangle of the co-relation matrix
mask = np.tril(np.ones_like(sequence_matrix))

fig, axes = plt.subplots(1, 2, figsize=(24, 8))
im = sns.heatmap(np.triu(sequence_matrix), cmap='viridis', xticklabels=sorted_cities, mask=mask,
                 linecolor='k', cbar_kws={'label': 'Migration'}, ax = axes[0], annot=True, fmt=".0f")
cbar = im.collections[0].colorbar
cbar.set_label('Migration', size=12.5, labelpad=5)
cbar.ax.yaxis.set_label_position('left')
axes[0].set_yticks(ticks=np.arange(len(sorted_cities))+.5, labels=sorted_cities,
                   rotation=360)
axes[0].set_title('Migration between cities', fontsize=20)
axes[0].tick_params(axis='both', labelsize=12)

# Create the scatter plot with a custom legend
legend_handles = []  # Collect legend handles
legend_labels = []   # Collect legend labels

for i in range(len(cities)):
    if labels[i] not in legend_labels:
        legend_handles.append(axes[1].scatter([], [], color=colors[i], edgecolors='black', marker=markers[i], s=sizes[i]))
        legend_labels.append(labels[i])  # Add marker label to legend labels list
    axes[1].scatter(X[i], Y[i], color=colors[i], edgecolors='black', marker=markers[i], s=sizes[i])
# axes[1].scatter(X, Y, color=colors, edgecolors='black', marker=markers, s=150)
axes[1].set_title("Coordinates of cities", fontsize=20)
axes[1].tick_params(axis='both', labelsize=12.5)
axes[1].set_facecolor('LightGrey')
include_river = False
if include_river:
    axes[1].axhline(40, xmin=0, xmax=0.55, linewidth=15, color='aqua')
    axes[1].axvline(40, ymin=0, ymax=0.45, linewidth=15, color='aqua')
    axes[1].axhline(40, xmin=0.5, xmax=0.62, linewidth=15, color='aqua')
    axes[1].axvline(50, ymin=0.455, ymax=0.63, linewidth=15, color='aqua')
    axes[1].axhline(60, xmin=0.635, xmax=0.745, linewidth=15, color='aqua')
    axes[1].axvline(60, ymin=0.66, ymax=1, linewidth=15, color='aqua')
axes[1].set_xlabel('Latitude', fontsize=15)
axes[1].set_ylabel('Longitude', fontsize=15)
# Add the legend to the scatter plot
axes[1].legend(handles=legend_handles, labels=legend_labels, loc='lower right', fontsize=12)
# axes[1].set_xlim(0,)
# axes[1].set_ylim(0,)
for i, (x, y) in enumerate(zip(X, Y)):
    axes[1].text(x+1.25, y+1.25, cities[i], fontsize=text_sizes[i], color='k')
axes[1].grid(True, color='black', linestyle='dotted')
fig.suptitle("West-East Travel & City Locations", fontsize=25)
# plt.tight_layout()
if not os.path.exists(r"algorithm_repository/Flow-Maps/data_vis.pdf"):
    plt.savefig(r"algorithm_repository/Flow-Maps/data_vis.pdf", bbox_inches="tight")
plt.show()

# Convert data to dataframe
sequence_df = pd.DataFrame(sequence_matrix, index=sorted_cities, columns=sorted_cities)
if not(os.path.exists(r"algorithm_repository/Flow-Maps/data.csv")):
    sequence_df.to_csv(r"algorithm_repository/Flow-Maps/data.csv")

fig, axes = plt.subplots(1, 1, figsize=(15, 8))
im = sns.heatmap(np.triu(sequence_matrix), cmap='viridis', xticklabels=sorted_cities, mask=mask,
                 linecolor='k', cbar_kws={'label': 'Migration'}, ax = axes, annot=True, fmt=".0f")
cbar = im.collections[0].colorbar
cbar.set_label('Migration', size=15, labelpad=5)
cbar.ax.yaxis.set_label_position('left')
axes.set_yticks(ticks=np.arange(len(sorted_cities))+.5, labels=sorted_cities,
                   rotation=360)
axes.set_title('Migration between cities', fontsize=20)
axes.tick_params(axis='both', labelsize=15)
if not os.path.exists(r"algorithm_repository/Flow-Maps/data_more_heatmap.pdf"):
    plt.savefig(r"algorithm_repository/Flow-Maps/data_more_heatmap.pdf", bbox_inches="tight")
plt.show()

fig, axes = plt.subplots(1, 1, figsize=(15, 8))
# Create the scatter plot with a custom legend
legend_handles = []  # Collect legend handles
legend_labels = []   # Collect legend labels

for i in range(len(cities)):
    if labels[i] not in legend_labels:
        legend_handles.append(axes.scatter([], [], color=colors[i], edgecolors='black', marker=markers[i], s=sizes[i]))
        legend_labels.append(labels[i])  # Add marker label to legend labels list
    axes.scatter(X[i], Y[i], color=colors[i], edgecolors='black', marker=markers[i], s=sizes[i])
# axes[1].scatter(X, Y, color=colors, edgecolors='black', marker=markers, s=150)
axes.set_title("Coordinates of cities", fontsize=20)
axes.tick_params(axis='both', labelsize=12.5)
axes.set_facecolor('LightGrey')
include_river = False
if include_river:
    axes.axhline(40, xmin=0, xmax=0.55, linewidth=15, color='aqua')
    axes.axvline(40, ymin=0, ymax=0.45, linewidth=15, color='aqua')
    axes.axhline(40, xmin=0.5, xmax=0.62, linewidth=15, color='aqua')
    axes.axvline(50, ymin=0.455, ymax=0.63, linewidth=15, color='aqua')
    axes.axhline(60, xmin=0.635, xmax=0.745, linewidth=15, color='aqua')
    axes.axvline(60, ymin=0.66, ymax=1, linewidth=15, color='aqua')
axes.set_xlabel('Latitude', fontsize=15)
axes.set_ylabel('Longitude', fontsize=15)
# Add the legend to the scatter plot
axes.legend(handles=legend_handles, labels=legend_labels, loc='lower right', fontsize=12)
# axes[1].set_xlim(0,)
# axes[1].set_ylim(0,)
for i, (x, y) in enumerate(zip(X, Y)):
    axes.text(x+1.25, y+1.25, cities[i], fontsize=text_sizes[i], color='k')
axes.grid(True, color='black', linestyle='dotted')
# plt.tight_layout()
if not os.path.exists(r"algorithm_repository/Flow-Maps/data_loc.pdf"):
    plt.savefig(r"algorithm_repository/Flow-Maps/data_loc.pdf", bbox_inches="tight")
plt.show()