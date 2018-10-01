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

# model = Generator(103, 9)
# model.load_state_dict(torch.load("C:\\Users\mmyller.KPLABS\PycharmProjects\software\python_research\\augmentation\GAN_1\\last_model", map_location='cpu'))
#
# pca = PCA(2)
# pca2 = PCA(2)
# dataset = HyperspectralDataset("C:\\Users\mmyller.KPLABS\Documents\datasets\pavia\PaviaU_corrected.npy", "C:\\Users\mmyller.KPLABS\Documents\datasets\pavia\PaviaU_gt.npy", normalize=True)
# fake = np.zeros(dataset.x.shape)
# # dataset.x = pca.fit_transform(dataset.x)
# classes = np.unique(dataset.y)
#
# real_label_samples = dict.fromkeys(classes)
# for label in real_label_samples:
#     indexes = np.where(dataset.y == label)[0]
#     samples = dataset.x[indexes]
#     shuffle(samples)
#     # samples = samples[:20]
#     real_label_samples[label] = samples
#
# fake_label_samples = dict.fromkeys(classes)
# last_insert = 0
# for label in fake_label_samples:
#     samples = []
#     noise = torch.FloatTensor(np.random.normal(0.5, 0.1, (len(real_label_samples[label]), 103)))
#     noise = torch.cat([noise, torch.tensor(to_categorical(np.full(len(real_label_samples[label]) ,label), 9)).type(torch.FloatTensor)], dim=1)
#     fake_samples = model(noise)
#     fake[last_insert:last_insert+len(fake_samples)] = fake_samples.detach().numpy()
#     last_insert = last_insert+len(fake_samples)
# last = 0
# # fake = pca2.fit_transform(fake)
# for label in fake_label_samples:
#     fake_label_samples[label] = fake[last:last + len(real_label_samples[label])]
#     last = last + len(real_label_samples[label])
# label = 1
# colors = get_cmap(16)
# real_label_average = np.mean(real_label_samples[label], axis=0)
# fake_label_average = np.mean(fake_label_samples[label], axis=0)
# real_label_std = np.std(real_label_samples[label], axis=0)
# fake_label_std = np.std(fake_label_samples[label], axis=0)
# x = [i for i in range(0,103)]
# plt.errorbar(x, real_label_average, real_label_std, c=colors(1))
# plt.errorbar(x, fake_label_average, fake_label_std, c=colors(11), alpha=0.5)


def plot_distribution(dataset, generator_model_path, output_path, input_shape, classes_count,
                      device='cpu'):
    model = Generator(input_shape, classes_count)
    if device == 'cpu':
        model.load_state_dict(torch.load(generator_model_path, map_location=device))
    else:
        model.load_state_dict(torch.load(generator_model_path))
    classes = np.unique(dataset.y)
    original_dataset_shape = dataset.x.shape
    real_label_samples = dict.fromkeys(classes)

    real_data = PCA(2).fit_transform(dataset.x)
    for label in real_label_samples:
        indexes = np.where(dataset.y == label)[0]
        samples = real_data[indexes]
        shuffle(samples)
        real_label_samples[label] = samples

    fake_dataset = generate_fake_dataset(model, input_shape, original_dataset_shape,
                                         real_label_samples, device=device)
    fake_dataset = PCA(2).fit_transform(fake_dataset)
    fake_label_samples = list_to_dict(fake_dataset, real_label_samples)

    colors = get_cmap(classes_count)
    for label in real_label_samples:
        plt.subplot(1, 2, 1)
        plt.scatter(real_label_samples[label][:, 0], real_label_samples[label][:, 1], s=1, c=colors(label))
        plt.title('Real')
    for label in real_label_samples:
        plt.subplot(1, 2, 2)
        plt.scatter(fake_label_samples[label][:, 0], fake_label_samples[label][:, 1], s=1, c=colors(label))
        plt.title('Fake')
    plt.savefig(output_path)


def generate_fake_dataset(model, input_shape, dataset_shape, real_label_samples, device='cpu'):
    fake = np.zeros(dataset_shape)
    fake_label_samples = dict.fromkeys(real_label_samples)
    classes_count = len(real_label_samples.keys())
    last_insert = 0
    for label in fake_label_samples:
        noise = torch.FloatTensor(np.random.normal(0.5, 0.1, (len(real_label_samples[label]), input_shape)))
        label_one_hot = to_categorical(np.full(len(real_label_samples[label]), label), classes_count)
        label_one_hot = torch.from_numpy(label_one_hot)
        if device == 'gpu':
            noise = noise.cuda()
            label_one_hot = label_one_hot.cuda()
        fake_samples = model(noise, label_one_hot)
        fake[last_insert:last_insert + len(fake_samples)] = fake_samples.detach().numpy()
        last_insert = last_insert + len(fake_samples)
    return fake


def list_to_dict(fake_dataset, real_label_samples):
    fake_label_samples = dict.fromkeys(real_label_samples.keys())
    last = 0
    for label in fake_label_samples:
        fake_label_samples[label] = fake_dataset[last:last + len(real_label_samples[label])]
        last = last + len(real_label_samples[label])
    return fake_label_samples
