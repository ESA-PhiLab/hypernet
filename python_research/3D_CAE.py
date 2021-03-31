"""
If you plan on using this implementation, please cite our work (https://ieeexplore.ieee.org/abstract/document/8948005):
@ARTICLE{Nalepa2020_3DCAE,
 author={J. {Nalepa} and M. {Myller} and Y. {Imai} and K. -I. {Honda} and T. {Takeda} and M. {Antoniak}},
 journal={IEEE Geoscience and Remote Sensing Letters},
 title={Unsupervised Segmentation of Hyperspectral Images Using 3-D Convolutional Autoencoders},
 year={2020},
 volume={17},
 number={11},
 pages={1948-1952},
 doi={10.1109/LGRS.2019.2960945}}
"""
import os
from typing import Tuple

import numpy as np
from time import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from skimage.io import imsave
from skimage.color import label2rgb
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

from python_research.io import save_to_csv
from python_research.dataset_structures import DataLoaderShuffle, HyperspectralCube


class DCEC(nn.Module):
    def __init__(self, input_dims: np.ndarray, n_clusters: int,
                 kernel_shape: np.ndarray, last_out_channels: int = 32, latent_vector_size: int = 25,
                 update_interval: int = 140, device: str='cpu',
                 artifacts_path: str='DCEC'):
        super(DCEC, self).__init__()
        self.latent_vector_size = latent_vector_size
        self.n_clusters = n_clusters
        self.last_out_channels = last_out_channels
        encoder_shape, out_features = self._calculate_shapes(input_dims, kernel_shape, 2,
                                                             self.last_out_channels)
        self.final_encoder_shape = tuple(np.hstack([self.last_out_channels, encoder_shape]))
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=kernel_shape),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv3d(in_channels=32, out_channels=self.last_out_channels,
                      kernel_size=kernel_shape),
            nn.ReLU(),
            Flatten(),
            nn.Linear(in_features=out_features, out_features=self.latent_vector_size)
        )
        self.linear = nn.Sequential(
            nn.Linear(in_features=self.latent_vector_size, out_features=out_features),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(in_channels=self.last_out_channels, out_channels=32,
                               kernel_size=kernel_shape),
            nn.ReLU(),
            nn.Dropout(),
            nn.ConvTranspose3d(in_channels=32, out_channels=1,
                               kernel_size=kernel_shape)
        )
        self.clustering_layer = ClusteringLayer(n_clusters=self.n_clusters,
                                                input_dim=self.latent_vector_size)
        self.log_softmax = nn.LogSoftmax()
        self.update_interval = update_interval
        self.n_clusters = n_clusters
        self.artifacts_path = artifacts_path
        self.mse_loss = nn.MSELoss()
        self.kld_loss = nn.KLDivLoss(reduction='batchmean')
        self.device = device
        self.metrics = {'MSE': [],
                        'KLD': [],
                        'NMI': [0],
                        'ARS': [0]}
        self._best_autoencoder_loss = np.inf
        self._best_nmi = -np.inf

    def _calculate_shapes(self, input_dims: np.ndarray, kernel_shapes: np.ndarray, kernels_count: int,
                          channels_count: int):
        final_encoder_shape = input_dims - ((kernels_count * kernel_shapes) - kernels_count)
        out_features = np.prod(final_encoder_shape) * channels_count
        return final_encoder_shape, out_features

    def predict_clusters(self, data_loader):
        predicted_clusters = torch.zeros((len(data_loader) *
                                        data_loader.batch_size, self.n_clusters), device=self.device)
        last_insert = 0
        with torch.no_grad():
            for batch_x in data_loader:
                batch_x = Variable(batch_x, requires_grad=False).float()
                predicted_clusters[last_insert:
                                   last_insert +
                                   data_loader.batch_size, :] = self.clustering_layer(self.encoder(batch_x))
                last_insert += data_loader.batch_size
        return predicted_clusters

    @staticmethod
    def calculate_target_distribution(q):
        weight = torch.pow(q, 2) / torch.sum(q, dim=0)
        return torch.t(torch.t(weight) / torch.sum(weight, dim=1))

    def get_target_distribution(self, data_loader: DataLoader):
        return self.calculate_target_distribution(
            self.predict_clusters(data_loader))

    def encode_features(self, data_loader: DataLoader):
        encoded_features = torch.zeros((len(data_loader) *
                                        data_loader.batch_size, self.latent_vector_size), device=self.device)
        last_insert = 0
        with torch.no_grad():
            for batch_x in data_loader:
                batch_x = Variable(batch_x, requires_grad=False).float()
                encoded_features[last_insert:
                                 last_insert +
                                 data_loader.batch_size, :] = self.encoder(batch_x)
                last_insert += data_loader.batch_size
        return encoded_features

    def initialize(self, data_loader):
        encoded_features = self.encode_features(data_loader)
        kmeans = KMeans(n_clusters=self.n_clusters).fit(encoded_features.cpu().detach())
        cluster_centers = kmeans.cluster_centers_.astype(np.float32)
        cluster_centers = torch.from_numpy(cluster_centers).to(device=self.device)
        self.clustering_layer.set_weights(cluster_centers)

    def train_with_clustering(self, data_loader, optimizer, iterations: int,
                              gamma: float):
        self.initialize(data_loader)
        last_batch = 0
        true_labels = (data_loader.data.labels
                       .cpu()
                       .detach()
                       .numpy()
                       .transpose()
                       .reshape(-1))
        for iteration in range(iterations):
            if iteration % self.update_interval == 0:
                last_batch = 0
                data_loader.sort()
                predicted_labels = self.cluster_with_model(data_loader)
                self.plot_high_res(predicted_labels, data_loader.cube_2d_shape(),
                          iteration, 'model')
                self.metrics['NMI'].append(self.calculate_nmi(true_labels,
                                                              predicted_labels))
                self.metrics['ARS'].append(self.calculate_ars(true_labels,
                                                              predicted_labels))
                self._log_metrics_to_file()
                self._print_losses(iteration)
                self._save_model(iteration)
                data_loader.shuffle()
                target_distribution = self.get_target_distribution(data_loader)
                iter(data_loader)
            optimizer.zero_grad()
            try:
                batch_x = next(data_loader)
                batch_x = Variable(batch_x).float()
            except StopIteration:
                iter(data_loader)
                last_batch = 0
                continue
            encoder_output = self.encoder(batch_x)
            clustering_layer_output = self.log_softmax(self.clustering_layer(encoder_output))
            div_loss = self.kld_loss(clustering_layer_output,
                                     target_distribution[last_batch:
                                                         last_batch +
                                                         data_loader.batch_size]) * gamma
            linear_output = self.linear(encoder_output)
            linear_output = torch.reshape(linear_output,
                                          ((data_loader.batch_size, ) + self.final_encoder_shape))
            decoder_output = self.decoder(linear_output)
            mse_loss = self.mse_loss(batch_x, decoder_output)
            self.metrics['MSE'].append(mse_loss.item())
            self.metrics['KLD'].append(div_loss.item())
            div_loss.backward(retain_graph=True)
            mse_loss.backward()
            optimizer.step()
            last_batch += data_loader.batch_size

    def train_autoencoder(self, data_loader, optimizer, epochs, epsilon):
        true_labels = (data_loader.data.labels
                       .cpu()
                       .detach()
                       .numpy()
                       .transpose()
                       .reshape(-1))
        last_mse = 0
        for epoch in range(epochs):
            data_loader.shuffle()
            for batch_x in data_loader:
                batch_x = Variable(batch_x).float()
                optimizer.zero_grad()
                encoded = self.encoder(batch_x)
                linear_output = self.linear(encoded)
                reshaped = torch.reshape(linear_output,
                                         ((data_loader.batch_size, ) + self.final_encoder_shape))
                decoder_output = self.decoder(reshaped)
                mse_loss = self.mse_loss(batch_x, decoder_output)
                self.metrics['MSE'].append(mse_loss.item())
                mse_loss.backward()
                optimizer.step()
            data_loader.sort()
            self.save_if_best(np.average(self.metrics['MSE']))
            # predicted_labels = self.cluster_with_kmeans(data_loader)
            predicted_labels = self.cluster_with_gaussian(data_loader)
            self.plot_high_res(predicted_labels, data_loader.cube_2d_shape(),
                      epoch, 'kmeans')
            self.metrics['NMI'].append(self.calculate_nmi(true_labels,
                                                          predicted_labels))
            self.metrics['ARS'].append(self.calculate_ars(true_labels,
                                                          predicted_labels))
            self._log_metrics_to_file()
            if epoch > 1:
                if np.abs(last_mse - np.average(self.metrics['MSE'])) < epsilon:
                    break
            last_mse = np.average(self.metrics['MSE'])
            self._print_losses(epoch)

    @staticmethod
    def calculate_nmi(labels_true, labels_predicted):
        to_delete = np.where(labels_true == 0)[0]
        labels_predicted = np.delete(labels_predicted, to_delete)
        labels_true = np.delete(labels_true, to_delete).astype(np.int32)
        return normalized_mutual_info_score(labels_true, labels_predicted)

    @staticmethod
    def calculate_ars(labels_true, labels_predicted):
        to_delete = np.where(labels_true == 0)[0]
        labels_predicted = np.delete(labels_predicted, to_delete)
        labels_true = np.delete(labels_true, to_delete)
        return adjusted_rand_score(labels_true, labels_predicted)

    def _print_losses(self, iteration):
        print('Iter: {}, MSE -> {} KLD -> {}, NMI -> {}, ARS -> {}'
              .format(iteration,
                      np.average(self.metrics['MSE']),
                      np.average(self.metrics['KLD'])
                                    if len(self.metrics['KLD']) != 0 else 0,
                      self.metrics['NMI'][-1],
                      self.metrics['ARS'][-1]))
        self.metrics['KLD'] = []
        self.metrics['MSE'] = []

    def _log_metrics_to_file(self):
        path = os.path.join(self.artifacts_path, 'metrics.csv')
        save_to_csv(path, [np.average(self.metrics['MSE']),
                           np.average(self.metrics['KLD']),
                           self.metrics['NMI'][-1],
                           self.metrics['ARS'][-1]])

    def save_if_best(self, loss):
        if loss < self._best_autoencoder_loss:
            self._save_model()
            self._best_autoencoder_loss = loss

    def train_model(self, data_loader, optimizer, epochs: int=200,
                    iterations: int=10000, gamma: float=0.1, epsilon=0.00001):
        print("Pretraining autoencoder:")
        training_start = time()
        self.train_autoencoder(data_loader, optimizer, epochs, epsilon)
        print("Pretraining finished, training with clustering")
        self.load_state_dict(torch.load(os.path.join(self.artifacts_path,
                                                     "best_autoencoder_model.pt")))
        self.train_with_clustering(data_loader, optimizer, iterations, gamma)
        training_time = time() - training_start
        save_to_csv(os.path.join(self.artifacts_path, "time.csv"), [training_time])
        print("Done!")

    def cluster_with_kmeans(self, data_loader):
        encoded_features = self.encode_features(data_loader)
        return KMeans(n_clusters=self.n_clusters).fit_predict(
            encoded_features.cpu().detach())

    def cluster_with_gaussian(self, data_loader):
        encoded_features = self.encode_features(data_loader)
        return GaussianMixture(n_components=self.n_clusters).fit_predict(
            encoded_features.cpu().detach())

    def cluster_with_model(self, data_loader):
        clusters = self.predict_clusters(data_loader)
        return np.argmax(clusters.cpu().detach().numpy(), axis=1)

    def plot_high_res(self, predicted_labels, shape_2d: Tuple[int, int], epoch: int,
                      clustering_method_name: str, true_labels=None):
        # if true_labels is not None:
        #     predicted_labels[true_labels == 0] = 11
        labels = predicted_labels.reshape(np.flip(shape_2d, axis=0))
        labels = labels.transpose()
        labels = label2rgb(labels)
        dir = os.path.join(self.artifacts_path, clustering_method_name
                           + '_clustering')
        os.makedirs(dir, exist_ok=True)
        file_name = os.path.join(dir, "plot_{}.png".format(epoch))
        imsave(file_name, labels)

    def _save_model(self, epoch: int = None):
        os.makedirs(self.artifacts_path, exist_ok=True)
        if epoch is not None:
            path = os.path.join(self.artifacts_path, "model_epoch_{}.pt".format(epoch))
        else:
            path = os.path.join(self.artifacts_path, "best_autoencoder_model.pt")
        torch.save(self.state_dict(), path)

    def forward(self, *input):
        pass


class Flatten(nn.Module):
    def forward(self, input_data):
        return input_data.view(input_data.size(0), -1)


class ClusteringLayer(nn.Module):
    def __init__(self, n_clusters: int, input_dim: int):
        super(ClusteringLayer, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(n_clusters, input_dim))
        nn.init.xavier_normal_(self.weights)

    def set_weights(self, weights):
        self.weights = nn.Parameter(weights).float()

    def forward(self, input_data):
        q = 1.0 / (1.0 + (torch.sum(torch.pow(input_data.unsqueeze(dim=1) -
                                              self.weights, 2), dim=2)))
        q = torch.t(torch.t(q) / torch.sum(q, dim=1))
        return q


if __name__ == '__main__':
    device = 'cuda:0'
    out_path = r""
    # Example for the Houston dataset
    dataset_bands = 50
    neighborhood_size = 5
    epochs = 25

    dataset_height = 1202
    dataset_width = 4768
    samples_count = dataset_height * dataset_width

    batch_size = 596 # The batch size has to be picked in such a way that samples_count % batch_size == 0
    update_interval = int(samples_count / batch_size)
    iterations = int(update_interval * epochs) # This indicates the number of epochs that the clustering part of the autoencoder will be trained for

    dataset = HyperspectralCube(r"", # Path to .npy file or np.ndarray with [HEIGHT, WIDTH, BANDS] dimensions
                                neighbourhood_size=neighborhood_size,
                                device=device, bands=dataset_bands)
    dataset.standardize()
    dataset.convert_to_tensors(device=device)
    # Train
    net = DCEC(input_dims=np.array([dataset_bands, neighborhood_size, neighborhood_size]), n_clusters=20,
               kernel_shape=np.array([5, 3, 3]), latent_vector_size=20,
               update_interval=update_interval, device=device,
               artifacts_path=out_path)
    net = net.cuda(device=device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    data_loader = DataLoaderShuffle(dataset, batch_size=batch_size)
    net.train_model(data_loader, optimizer, epochs=epochs, iterations=iterations, gamma=0.1)

    # Predict
    data_loader.sort()
    net.load_state_dict(torch.load(out_path + "/model_path.pt"))
    predicted_labels = net.cluster_with_model(data_loader)
    net.plot_high_res(predicted_labels, dataset.original_2d_shape, -1, "model")
