import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_classification
from sklearn.svm import SVC, LinearSVC

# Generate synthetic dataset with increased variance and dispersion
X, y = make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=2, random_state=44, class_sep=1, flip_y = 0.1)


# Train classifiers
linear_svc = LinearSVC(random_state=42)
linear_svc.fit(X, y)

rbf_svc = SVC(kernel='rbf', gamma='scale', random_state=42)
rbf_svc.fit(X, y)

# Plot decision boundaries
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

Z_linear = linear_svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z_linear = Z_linear.reshape(xx.shape)

Z_rbf = rbf_svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z_rbf = Z_rbf.reshape(xx.shape)

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

plt.figure(figsize=(15, 6))

# Plot Linear SVC decision boundary
plt.subplot(1, 2, 1)
plt.pcolormesh(xx, yy, Z_linear, cmap=cmap_light)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title("Linear SVC Decision Boundary")

# Plot RBF SVC decision boundary
plt.subplot(1, 2, 2)
plt.pcolormesh(xx, yy, Z_rbf, cmap=cmap_light)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title("RBF SVC Decision Boundary")

plt.tight_layout()
plt.show()
