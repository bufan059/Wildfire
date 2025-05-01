

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pygments.lexers.srcinfo import keywords

data = np.load('/Users/bufan99/Desktop/ucr/25spring/project/autokg/KG_data1/KGtest0427.npy', allow_pickle=True).item()

print("Keys in data:", data.keys())

# adjacency matrix
adj_matrix = data['A']
print(adj_matrix.shape)

# keywords
# kw = data['keywords']
# print("keywords: ", kw)

# texts
# texts = data['texts']
# print("texts: ", texts)
# print(len(texts))


# create graph
G = nx.from_numpy_array(adj_matrix)

# visualize
plt.figure(figsize=(8, 6))
nx.draw(G, with_labels=True, node_color='skyblue', edge_color='gray', node_size=500)

labels = {i: kw for i, kw in enumerate(data['keywords'])}
plt.figure(figsize=(10, 8))
nx.draw(G, labels=labels, node_color='lightgreen', edge_color='gray', node_size=600, font_size=10)
plt.title("Knowledge Graph with Keyword Labels")

# plt.savefig('/Users/bufan99/Desktop/ucr/25spring/project/autokg/KG_data1/visualize_kg0427')

plt.show()
