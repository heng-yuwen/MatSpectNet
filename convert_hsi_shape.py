"""This file convert the spectral measurement in spectraldb to shape matrix described in paper section 3.4
""" 
import glob
import numpy
import pandas as pd
import numpy as np
csv_files = glob.glob("data/spectraldb/*.csv")
print("There are {} files in the spectraldb.".format(len(csv_files)))

def cal_shape_matrix(spectra):
    length = len(spectra)
    matrix = np.zeros((length, length))
    for i in range(length):
        for j in range(length):
            matrix[i, j] = abs(spectra[i] - spectra[j])
    
    return matrix

# organise the shape metrics in a tensor array.
shape_metrics = []
sampled_wavelengths = list(range(400, 701, 10)) # 31 wavelengths
hsi_shape_queries = np.zeros((len(csv_files), 465))
idx = 0
for csv_file in csv_files:
    pd_csv = pd.read_csv(csv_file, header=0, index_col=0)
    spectra = pd_csv["sce"].loc[sampled_wavelengths].values / 100
    length = 30
    end = 30
    for i in range(1, 31):
        hsi_shape_queries[idx, end-length:end] = abs(spectra[i-1:30] - spectra[i:])
        length -= 1
        end += length
    
    idx += 1

# save the converted shape metrics
# shape_metrics = np.array(shape_metrics)
np.save("./data/spectraldb/shape_metrics.npy", hsi_shape_queries)
