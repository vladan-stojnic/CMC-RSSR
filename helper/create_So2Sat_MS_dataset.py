import h5py
import lmdb
import numpy as np


def dumps_pyarrow(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    import pyarrow as pa
    
    return pa.serialize(obj).to_buffer()


SO2SAT_PATH = 'SO2SAT_TRAIN_HDF5_PATH'
OUTPUT_PATH = 'OUTPUT_LMDB_PATH'

fid = h5py.File(SO2SAT_PATH, 'r')
data = np.array(fid['sen2'])
labels = np.array(fid['label'])

db = lmdb.open(OUTPUT_PATH, map_size = 1e13)

txn = db.begin(write = True)

patch_names = []

for idx in range(data.shape[0]):
    print(idx)
    img = data[idx, :, :, :]
    label = labels[idx, :].argmax()

    image_name = 'train_'+str(idx)+'.png'

    txn.put(u'{}'.format(image_name).encode('ascii'), dumps_pyarrow((img, label)))
    patch_names.append(image_name)

    if idx % 5000 == 0:
        txn.commit()
        txn = db.begin(write = True)

txn.commit()
txn = db.begin(write = True)

SO2SAT_PATH = 'SO2SAT_VAL_HDF5_PATH'

fid = h5py.File(SO2SAT_PATH, 'r')
data = np.array(fid['sen2'])
labels = np.array(fid['label'])

for idx in range(data.shape[0]):
    print(idx)
    img = data[idx, :, :, :]
    label = labels[idx, :].argmax()

    image_name = 'val_'+str(idx)+'.png'

    txn.put(u'{}'.format(image_name).encode('ascii'), dumps_pyarrow((img, label)))
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
