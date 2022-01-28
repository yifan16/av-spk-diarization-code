from sklearn.datasets import load_digits
from MulticoreTSNE import MulticoreTSNE as TSNE
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
pca = PCA(n_components=2)
import pdb
#pdb.set_trace()
#digits = load_digits()
imfeat = np.load('im.npy')
ccfeat = np.load('cc.npy')
#embeddings = TSNE(n_jobs=4).fit_transform(imfeat)
#embeddings = pca.fit(imfeat).transform(imfeat)
embeddings = pca.fit_transform(imfeat)
vis_x = embeddings[:, 0]
vis_y = embeddings[:, 1]

#plt.scatter(vis_x, vis_y, c=digits.target, cmap=plt.cm.get_cmap("jet", 10), marker='.')
plt.scatter(vis_x, vis_y,  cmap=plt.cm.get_cmap("jet", 10), marker='.')
#plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)
plt.show()

#embeddings = TSNE(n_jobs=4).fit_transform(ccfeat)
pca = PCA(n_components=2)
#embeddings = pca.fit(imfeat).transform(imfeat)
embeddings = pca.fit_transform(ccfeat)
vis_x = embeddings[:, 0]
vis_y = embeddings[:, 1]

#plt.scatter(vis_x, vis_y, c=digits.target, cmap=plt.cm.get_cmap("jet", 10), marker='.')
plt.scatter(vis_x, vis_y,  cmap=plt.cm.get_cmap("jet", 10), marker='.')
#plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)
plt.show()