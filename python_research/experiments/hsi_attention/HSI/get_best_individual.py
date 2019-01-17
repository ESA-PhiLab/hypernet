import os

import numpy as np

path = r'C:\Users\ltulczyjew\Desktop\bombs_results\pavia\9_bands'
correlation_to_attention = \
    r'C:\Users\ltulczyjew\Desktop\software\python_research\experiments\attention_based_selected_bands\pavia\0.01.txt'

ranges = list(range(1, 31))
dist, entr, bands = [], [], []
for i in ranges:
    dist.append(np.loadtxt(os.path.join(path, 'best_individual_distance_' + str(i)), dtype=float).astype(float))
    entr.append(np.loadtxt(os.path.join(path, 'best_individual_entropy_' + str(i)), dtype=float).astype(float))
    bands.append(np.loadtxt(os.path.join(path, 'best_individual_bands_' + str(i)), dtype=int).astype(int))

i = np.argmax(dist).astype(int)
dist = list(map(float, dist))
entr = list(map(float, entr))

entropy = entr[i]
distance = dist[i]
bands = bands[i].astype(int).tolist()
if not os.path.exists(os.path.join(path, 'final')):
    os.mkdir(os.path.join(path, 'final'))
np.savetxt(os.path.join(path, 'final', 'bombs_selected_bands'), bands, fmt='%d')
np.savetxt(os.path.join(path, 'final', 'bombs_selected_entropy'), np.asarray(entropy).reshape(1, 1), fmt='%2.3f')
np.savetxt(os.path.join(path, 'final', 'bombs_selected_distance'), np.asarray(distance).reshape(1, 1), fmt='%2.3f')
attention = np.sort(np.loadtxt(correlation_to_attention)).astype(int).tolist()
corr = np.corrcoef(bands, attention)
print(corr)
np.savetxt(os.path.join(path, 'final', 'bombs_correlation_to_attention'), corr, fmt='%2.3f')
pass
