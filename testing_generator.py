import torch
from python_research.augmentation.generator import Generator
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from python_research.augmentation.dataset import HyperspectralDataset
from random import shuffle

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

model = Generator(201)
model.load_state_dict(torch.load("C:\\Users\MMyller\PycharmProjects\software\generator_model", map_location='cpu'))

dataset = HyperspectralDataset("C:\\Users\MMyller\Documents\datasets\Indian\Indian_pines_corrected.npy", "C:\\Users\MMyller\Documents\datasets\Indian\Indian_pines_gt.npy", normalize=False)

classes = np.unique(dataset.y)
classes = classes

real_label_samples = dict.fromkeys(classes)
for label in real_label_samples:
    indexes = np.where(dataset.y == label)[0]
    samples = dataset.x[indexes]
    shuffle(samples)
    # samples = samples[:20]
    real_label_samples[label] = samples

fake_label_samples = dict.fromkeys(classes)
for label in fake_label_samples:
    samples = []
    noise = torch.FloatTensor(np.random.normal(0, 1, (len(real_label_samples[label]), 200)))
    noise = torch.cat([noise, torch.tensor(np.full((len(real_label_samples[label]), 1), label)).type(torch.FloatTensor)], dim=1)
    fake_samples = model(noise)
    fake_label_samples[label] = fake_samples

pca = PCA(2)

for label in real_label_samples:
    real_label_samples[label] = pca.fit_transform(real_label_samples[label])
    fake_label_samples[label] = pca.fit_transform(fake_label_samples[label].detach().numpy())
colors = get_cmap(16)
for label in real_label_samples:
    plt.subplot(1, 2, 1)
    plt.scatter(real_label_samples[label][:, 0], real_label_samples[label][:, 1], s=1, c=colors(label))
    plt.title('Real')
for label in real_label_samples:
    plt.subplot(1, 2, 2)
    plt.scatter(fake_label_samples[label][:, 0], fake_label_samples[label][:, 1], s=1, c=colors(label))
    plt.title('Fake')
plt.show()
a=5