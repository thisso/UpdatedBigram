import gensim.downloader as api
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Load GloVe embeddings
model = api.load("glove-wiki-gigaword-50")

# Words to visualize
words = ["woman", "man", "aunt","uncle"]

# Get vectors
vectors = np.array([model[word] for word in words])

# Reduce to 3D using PCA
pca = PCA(n_components=3)
vectors_3d = pca.fit_transform(vectors)

# Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot lines (vectors) from origin to each point
for vec, word in zip(vectors_3d, words):
    x, y, z = vec
    ax.plot([0, x], [0, y], [0, z], color='blue')  # Line from origin
    ax.scatter(x, y, z, color='red')               # Dot at tip
    ax.text(x, y, z, word, fontsize=10)            # Label

ax.set_title("3D Word Embedding Vectors (GloVe + PCA)")
plt.show()
