from PIL import Image
import numpy as np
import tifffile
import re
import cv2
import matplotlib.pyplot as plt
import os
from itertools import groupby
import subprocess

def find_files(folder, types):
    return [entry.path for entry in os.scandir(folder) if(entry.is_file() and os.path.splitext(entry.name)[1].lower() in types)]

def sort_key(item):
    band_order = {'G': 0, 'NIR': 1, 'R': 2, 'RE': 3}
    image_number_str = item.split('DJI_')[-1].split('_')[1]
    image_number = int(image_number_str)
    # print(image_number, item)
    band_type = item.split("_")[-1].split('.')[0]  # Extracting band type
    return image_number, band_order.get(band_type, float('inf'))  # Handling unexpected band types


def resolution_clip(rgb, ms, outfolder):
    rgb_image_orig = cv2.imread(rgb) 
    rgbname = os.path.basename(rgb)
    ms_images = [cv2.imread(m, cv2.IMREAD_UNCHANGED)  for m in ms]
    scale_x = 0.0032765 / 0.0021991
    scale_y = 0.0032861 / 0.0022119
    for mpath, m in zip(ms, ms_images):
        mname = os.path.basename(mpath)
        ms_resized = cv2.resize(m, None, fx=scale_x, fy=scale_y)
        cv2.imwrite(os.path.join(outfolder, mname), ms_resized)

        with tifffile.TiffFile(mpath) as tif:
            tif_tags = {}
            tif_tags['XMP'] = tif.pages[0].tags[700].value.decode('utf-8')
            s = open("meta.xml", 'w', encoding='utf-8')
            s.write(tif_tags['XMP'])
        command0 = f'"C:/exiftool(-k).exe" "-xmp<=C:/Users/User/Desktop/Fizza/MultiSpectral-Image-Correction/meta.xml" "{os.path.join(outfolder, mname)}"'
        process = subprocess.Popen(command0, shell=True, stdin=subprocess.PIPE)
        process.communicate(input=b'\n')
        command1 = f'"C:/exiftool(-k).exe" -tagsfromfile "{rgb}" -r -GPSPosition -GPSLongitude -GPSLatitude -GPSAltitude -FocalLength -FieldOfView -xmp:all "{os.path.join(outfolder, mname)}"'  
        process = subprocess.Popen(command1, shell=True, stdin=subprocess.PIPE)
        process.communicate(input=b'\n')


def main(input_folder,output_folder):
    ms_photos = find_files(input_folder, [".tif"])
    rgb_photos = find_files(input_folder, [".jpg"])
    sorted_data = sorted(ms_photos, key=sort_key)
    grouped = [list(g) for _, g in groupby(sorted_data, key=lambda x: x.split('DJI_')[-1].split('_')[1])]
    for i in range(len(grouped)):
        resolution_clip(rgb_photos[i], grouped[i], output_folder)


if __name__ == '__main__':
    input_folder = r"C:\Users\User\Desktop\Fizza\MultiSpectral-Image-Correction\set"
    output_folder = r"C:\Users\User\Desktop\Fizza\MultiSpectral-Image-Correction\outputset"
    main(input_folder,output_folder)