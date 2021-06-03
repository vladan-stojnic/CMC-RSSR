import json
import os

import lmdb
import numpy as np
import rasterio


def dumps_pyarrow(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    import pyarrow as pa
    
    return pa.serialize(obj).to_buffer()


BEN_ROOT = 'PATH_WITH_BIG_EARTH_NET_IMAGES'
OUTPUT_PATH = 'PATH_TO_SAVE_LMDB'
MAPPING_PATH = '../image_target_mapping/BigEarthNet_mapping.json'

with open(MAPPING_PATH, 'r') as f:
    mapping = json.load(f)

db = lmdb.open(OUTPUT_PATH, map_size = 1e13)

txn = db.begin(write = True)

patch_names = []

for idx, image_name in enumerate(mapping):
    print(idx)

    path_to_images = os.path.join(BEN_ROOT, image_name.split('.')[0])
    data_120 = []
    data_60 = []

    for band in ['B02', 'B03', 'B04', 'B08']:
        image_path = path_to_images + '/' + image_name.split('.')[0] + '_' + band + '.tif'
        band_ds = rasterio.open(image_path)
        band_data = band_ds.read(1)
        data_120.append(band_data)

    for band in ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12']:
        image_path = path_to_images + '/' + image_name.split('.')[0] + '_' + band + '.tif'
        band_ds = rasterio.open(image_path)
        band_data = band_ds.read(1)
        data_60.append(band_data)

    data_120 = np.asarray(data_120).astype(np.float32)
    data_60 = np.asarray(data_60).astype(np.float32)
    label = mapping[image_name]

    txn.put(u'{}'.format(image_name).encode('ascii'), dumps_pyarrow((data_120, data_60, label)))
    patch_names.append(image_name)

    if idx % 5000 == 0:
        txn.commit()
        txn = db.begin(write = True)

txn.commit()
keys = [u'{}'.format(patch_name).encode('ascii') for patch_name in patch_names]

with db.begin(write = True) as txn:
    txn.put(b'__keys__', dumps_pyarrow(keys))
    txn.put(b'__len__', dumps_pyarrow(len(keys)))

print("Flushing database ...")
db.sync()
db.close()
