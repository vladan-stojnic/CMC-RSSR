import os

import h5py
import numpy as np
from skimage.io import imsave


label_mapping = ['compact_high-rise', 'compact_middle-rise', 'compact_low-rise', 
                 'open_high-rise', 'open_middle-rise', 'open_low-rise',
                 'lighweight_low-rise', 'large_low-rise', 'sparsley_built',
                 'heavy_industrial', 'dense_trees', 'scattered_trees',
                 'bush_scrub', 'low_plants', 'bare_rock_or_paved',
                 'bare_soil_or_sand', 'water']

PATH = 'SO2SAT_TRAIN_HDF5_PATH'

ROOT = 'OUTPUT_DIRECTORY_PATH'

fid = h5py.File(PATH, 'r')
data = np.array(fid['sen2'])
labels = np.array(fid['label'])
                 
label_count = [0]*17

for i in range(data.shape[0]):
    img = data[i, :, :, 2::-1]
    img /= img.max()
    label = labels[i, :].argmax()
    
    label_dir = os.path.join(ROOT, label_mapping[label])
    
    if not os.path.exists(label_dir):
        os.mkdir(label_dir)
        
    imsave(os.path.join(label_dir, 'train_'+str(label_count[label])+'.png'), img)
    label_count[label] += 1

PATH = 'SO2SAT_VAL_HDF5_PATH'

fid = h5py.File(PATH, 'r')
data = np.array(fid['sen2'])
labels = np.array(fid['label'])
                 
label_count = [0]*17

for i in range(data.shape[0]):
    img = data[i, :, :, 2::-1]
    img /= img.max()
    label = labels[i, :].argmax()
    
    label_dir = os.path.join(ROOT, label_mapping[label])
    
    if not os.path.exists(label_dir):
        os.mkdir(label_dir)
        
    imsave(os.path.join(label_dir, 'val_'+str(label_count[label])+'.png'), img)
    label_count[label] += 1
