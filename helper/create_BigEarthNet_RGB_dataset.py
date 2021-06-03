from __future__ import print_function

import argparse
import os

import numpy as np
from skimage import io
from skimage.util import img_as_ubyte


OUTPUT_PATH = 'OUTPUT_DIRECTORY_PATH'

parser = argparse.ArgumentParser(
    description='This script eliminates patches with seasonal snow and cloud&shadow cover')
parser.add_argument('-r', '--root_folder', dest='root_folder',
                    help='root folder path contains multiple patch folders')
parser.add_argument('-s', '--snow_file', dest='snow_file',
                    help='list of patches file for seasonal snow cover')
parser.add_argument('-c', '--cloud_file', dest='cloud_file',
                    help='list of patches file for cloud&shadow cover')
parser.add_argument('-e', '--snow_cloud_files', dest='snow_cloud_files', nargs='+',
                    help='list of patches files for seasonal snow and cloud&shadow cover')
args = parser.parse_args()

# Checks the existence of root folder of patches
if args.root_folder:
    if not os.path.exists(args.root_folder):
        print('ERROR: folder', args.root_folder, 'does not exist')
        exit()
else:
    print('ERROR: -r argument is required')
    exit()

# Checks the correctness of other arguments 
file_paths = []
if args.snow_cloud_files:
    if not (len(args.snow_cloud_files) == 2):
        print('ERROR: two csv files must be provided for -e option')
        exit()
    print('INFO: patches with both seasonal snow and cloud&shadow cover will be eliminated')
    for file_path in args.snow_cloud_files:
        file_paths.append(file_path)
elif args.snow_file:
    print('INFO: patches with only seasonal snow cover will be eliminated')
    file_paths.append(args.snow_file)
elif args.cloud_file:
    print('INFO: patches with only cloud&shadow cover will be eliminated')
    file_paths.append(args.cloud_file)
else:
    print('ERROR: one of -e, -s and -c arguments is required')
    exit()

# Checks the existence of required python packages
gdal_existed = rasterio_existed = False
try:
    import gdal
    gdal_existed = True
    print('INFO: GDAL package will be used to read GeoTIFF files')
except ImportError:
    try:
        import rasterio
        rasterio_existed = True
        print('INFO: rasterio package will be used to read GeoTIFF files')
    except ImportError:
        print('ERROR: please install either GDAL or rasterio package to read GeoTIFF files')
        exit()
try:
    import csv
except ImportError:
    print('ERROR: please install csv package to read csv files')
    exit()

# Checks the existence of csv files and populate the list of patches which will be eliminated
elimination_patch_list = []  
for file_path in file_paths:
    if not os.path.exists(file_path):
        print('ERROR: file located at', file_path, 'does not exist')
        exit()
    with open(file_path, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            elimination_patch_list.append(row[0])
print('INFO:', len(elimination_patch_list), 'number of patches will be eliminated')
elimination_patch_list = set(elimination_patch_list)

# Spectral band names to read related GeoTIFF files
band_names = ['B04', 'B03', 'B02']

print(len(elimination_patch_list))

# Reads spectral bands of all patches except the ones in elimination list
for root, dirs, files in os.walk(args.root_folder):
    if not root == args.root_folder:
        patch_folder_path = root
        patch_name = os.path.basename(patch_folder_path)
        if not patch_name in elimination_patch_list:
            image = []
            for band_name in band_names:
                # First finds related GeoTIFF path and reads values as a numpy array
                band_path = os.path.join(
                    patch_folder_path, patch_name + '_' + band_name + '.tif')
                if gdal_existed:
                    band_ds = gdal.Open(band_path,  gdal.GA_ReadOnly)
                    raster_band = band_ds.GetRasterBand(1)
                    band_data = raster_band.ReadAsArray()
                elif rasterio_existed:
                    band_ds = rasterio.open(band_path)
                    band_data = band_ds.read(1)
                image.append(band_data)
                # band_data keeps the values of band band_name for the patch patch_name
                print('INFO: band', band_name, 'of patch', patch_name,
                        'is ready with size', band_data.shape)
            image = np.array(image)
            image = image.transpose((1, 2, 0))
            image = image.astype('float')
            image /= image.max()
            image = img_as_ubyte(image)
        
            io.imsave(os.path.join(OUTPUT_PATH, patch_name+'.tif'), image)
        else:
            print('INFO: patch', patch_name, 'is eliminated from reading')
