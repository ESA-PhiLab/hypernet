import torch
from python_research.augmentation.generator import Generator
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from python_research.augmentation.dataset import HyperspectralDataset
from random import shuffle
from keras.utils import to_categorical

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

model = Generator(200, 16)
model.load_state_dict(torch.load("python_research/augmentation/GAN_1/generator_model", map_location='cpu'))

pca = PCA(2)
pca2 = PCA(2)
dataset = HyperspectralDataset("C:\\Users\mmyller.KPLABS\Documents\datasets\Indian\Indian_pines_corrected.npy", "C:\\Users\mmyller.KPLABS\Documents\datasets\Indian\Indian_pines_gt.npy", normalize=True)
fake = np.zeros(dataset.x.shape)
dataset.x = pca.fit_transform(dataset.x)
classes = np.unique(dataset.y)

real_label_samples = dict.fromkeys(classes)
for label in real_label_samples:
    indexes = np.where(dataset.y == label)[0]
    samples = dataset.x[indexes]
    shuffle(samples)
    # samples = samples[:900]
    real_label_samples[label] = samples

fake_label_samples = dict.fromkeys(classes)
last_insert = 0
for label in fake_label_samples:
    samples = []
    noise = torch.FloatTensor(np.random.normal(0.5, 0.1, (len(real_label_samples[label]), 200)))
    labels = torch.FloatTensor(to_categorical(np.full(len(real_label_samples[label]), label), 16))
    fake_samples = model(noise, labels)
    fake[last_insert:last_insert+len(fake_samples)] = fake_samples.detach().numpy()
    last_insert = last_insert+len(fake_samples)
last = 0
fake = pca2.fit_transform(fake)
for label in fake_label_samples:
    fake_label_samples[label] = fake[last:last + len(real_label_samples[label])]
    last = last + len(real_label_samples[label])
label = 2
colors = get_cmap(16)
# real_label_average = np.mean(real_label_samples[label], axis=0)
# fake_label_average = np.mean(fake_label_samples[label], axis=0)
# real_label_std = np.std(real_label_samples[label], axis=0)
# fake_label_std = np.std(fake_label_samples[label], axis=0)
# x = [i for i in range(0,103)]
# plt.errorbar(x, real_label_average, real_label_std, c=colors(1))
# plt.errorbar(x, fake_label_average, fake_label_std, c=colors(11), alpha=0.5)

for label in real_label_samples:
    plt.subplot(1, 2, 1)
    plt.scatter(real_label_samples[label][:, 0], real_label_samples[label][:, 1], s=1, c=colors(label))
    plt.title('Real')
for label in real_label_samples:
    plt.subplot(1, 2, 2)
    plt.scatter(fake_label_samples[label][:, 0], fake_label_samples[label][:, 1], s=1, c=colors(label))
    plt.title('Fake')
plt.show()
