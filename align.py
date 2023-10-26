from PIL import Image
import numpy as np
import tifffile
import re
import cv2
import matplotlib.pyplot as plt
import os
from itertools import groupby
import subprocess
def img_overlaying(NIR_img, corrected_img):
    plt.imshow(cv2.addWeighted(cv2.imread(NIR_img,cv2.IMREAD_UNCHANGED), 0.5,cv2.imread(corrected_img,cv2.IMREAD_UNCHANGED), 0.5, 0))
def parse_value(text, keyword):
    pattern = re.escape(keyword) + r'\s*=\s*"([^"]*)"'
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        return None
def get_metadata(image_path):
    with tifffile.TiffFile(image_path) as tif:
        metadata = tif.pages[0].tags
        tif_tags = {}
        tif_tags['XMP'] = tif.pages[0].tags[700].value.decode('utf-8')
    s = tif_tags['XMP']
    # print(s)
    dwarp_data = "drone-dji:DewarpData"
    dwarp_data_para = parse_value(s, dwarp_data)
    dwarp_coefficients  = [float(x) for x in dwarp_data_para[11:].split(',')]
    calibrated_hmatrix = "drone-dji:DewarpHMatrix"
    calibrated_hmatrix = [float(x) for x in parse_value(s, calibrated_hmatrix).split(',')]
    # print(len(calibrated_hmatrix), calibrated_hmatrix)
    calibrated_hmatrix = np.array(calibrated_hmatrix).reshape((3,3))
    # print(calibrated_hmatrix)
    centre_x_para = 'drone-dji:CalibratedOpticalCenterX'
    centre_y_para = 'drone-dji:CalibratedOpticalCenterY'
    vignettingData_para = 'drone-dji:VignettingData'
    center_x = float(parse_value(s, centre_x_para))
    center_y = float(parse_value(s, centre_y_para))
    vignettingData_str = parse_value(s, vignettingData_para)
    vignetting_data = [float(x) for x in vignettingData_str.split(',')]  # Convert string to list of floats
    return center_x, center_y, vignetting_data, dwarp_coefficients, calibrated_hmatrix
def undistort_image(vignette_img, coefficients):
    center_x, center_y = coefficients[0], coefficients[1]
    fx, fy, cx, cy, k1, k2, p1, p2, k3 = coefficients[2]
    fx, fy, cx, cy, k1, k2, p1, p2, k3 = 2200.899902343750, 2200.219970703125, 10.609985351562, -6.575988769531, 0.008104680106, -0.042915198952, -0.000333522010, 0.000239991001, 0.000000000000
    matrix = np.array([[fx, 0, center_x+cx], [0, fy, center_y+cy], [0, 0, 1]])
    dist = np.array([k1, k2, p1, p2, k3])
    h, w = vignette_img.shape
    # img_norm = cv2.normalize( np.array(img), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # print(img_norm.max(), img_norm.min())
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(matrix, dist, (w,h), 1, (w,h))
    dst = cv2.undistort(np.array(vignette_img), matrix, dist, newcameramtx)
    return dst
    # print(np.array_equal(np.array(img), dst))
    # print(dst.max(), dst.min())
    # undistorted_image = Image.fromarray(dst)
    # cv2.imwrite(outpath_img, dst)

def apply_vignetting_correction(image_path,outpath_img):
    with Image.open(image_path) as img:
        width, height = img.size
        center_x, center_y, vignetting_data, dwarp_coefficients, calibrated_hmatrix = get_metadata(image_path)
        correction_img = np.zeros((height, width))
        for x in range(width):
            for y in range(height):
                r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                correction_value = sum([k * (r ** i) for i, k in enumerate((vignetting_data))]) + 1.0
                correction_img[y, x] = correction_value
        img_norm = cv2.normalize( np.array(img), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        corrected_img = img_norm * correction_img
        corrected_img = np.clip(corrected_img, 0, 255).astype(np.uint8)
    return (corrected_img, (center_x, center_y, dwarp_coefficients, calibrated_hmatrix))

def phase_alignment(dst_img, parameters):
    calibrated_hmatrix = parameters[-1]
    calibrated_hmatrix = np.array([[9.891065e-01, 1.740813e-02, -1.592078e+01],
                                   [-1.568817e-02, 9.885082e-01, 3.766531e+01],
                                   [1.083204e-06, 5.127963e-07, 1.000000e+00]])
    h, w = dst_img.shape
    transformed_image = cv2.warpPerspective(np.array(dst_img), calibrated_hmatrix, (w, h))
    return transformed_image

# def ecc_alignment(image1_path, image2_path, out_name):
#     # Read the images
#     image1 = cv2.imread(image1_path, cv2.IMREAD_UNCHANGED)
#     image2 = cv2.imread(image2_path, cv2.IMREAD_UNCHANGED)
#     # Check if images are read correctly
#     if image1 is None:
#         print(f"Error reading image from path: {image1_path}")
#         return
#     if image2 is None:
#         print(f"Error reading image from path: {image2_path}")
#         return
#     # Convert to grayscale if not already
#     if len(image1.shape) == 3:
#         image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
#     if len(image2.shape) == 3:
#         image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
#     # Ensure 8-bit unsigned
#     image1 = np.uint8(image1)
#     image2 = np.uint8(image2)
#     sift = cv2.SIFT_create()
#     # SIFT computation
#     keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
#     keypoints2, descriptors2 = sift.detectAndCompute(image2, None)
#     bf = cv2.BFMatcher()
#     matches = bf.knnMatch(descriptors1, descriptors2, k=2)
#         # Potential adjustment of the threshold, it's a tuning parameter.
#     threshold_ratio = 0.75
#     good_matches = [m for m, n in matches if m.distance < threshold_ratio * n.distance]
#     # Check if there are enough good matches
#     if len(good_matches) < 4:
#         print("Not enough good matches to compute homography")
#         return
#     good_matches = []
#     for m, n in matches:
#         if m.distance < 0.75 * n.distance:
#             good_matches.append(m)
#     matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#     src_points = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
#     dst_points = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
#     M, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
#     aligned_image2 = cv2.warpPerspective(image2, M, (image1.shape[1], image1.shape[0]))
#     print(np.array_equal(aligned_image2, image2))
#     cv2.imwrite(out_name, aligned_image2)


def find_files(folder, types):
    return [entry.path for entry in os.scandir(folder) if(entry.is_file() and os.path.splitext(entry.name)[1].lower() in types)]

def sort_key(item):
    band_order = {'G': 0, 'NIR': 1, 'R': 2, 'RE': 3}
    image_number_str = item.split('DJI_')[-1].split('_')[1]
    image_number = int(image_number_str)
    # print(image_number, item)
    band_type = item.split("_")[-1].split('.')[0]  # Extracting band type
    return image_number, band_order.get(band_type, float('inf'))  # Handling unexpected band types

import time
def pre_process_M3M(input_img,out_img,):
    st = time.time()
    vignette_img, parameters = apply_vignetting_correction(input_img,out_img)
    et = time.time()
    print("time elapsed in step 1", et-st)
    dst_img = undistort_image(vignette_img, parameters)
    et2 = time.time()
    print("time elapsed in step 2", et2-et)
    final_img = phase_alignment(dst_img, parameters)
    et3 = time.time()
    print("time elapsed in step 2", et3-et2)
    cv2.imwrite(out_img, final_img)
    # command = f'"C:/exiftool(-k).exe" -GpsLatitude="{latitude}" -GpsLongitude="{longitude}" -AbsoluteAltitude="{altitude}" -FlightPitchDegree="{pitch}" -FlightRollDegree="{roll}" -FlightYawDegree="{yaw}" "{d_file_path}"'
    command1 = f'"C:/exiftool(-k).exe" -tagsfromfile "{input_img}" -r -all:all -xmp:all -File:ImageSize "{out_img}"'
    process = subprocess.Popen(command1, shell=True, stdin=subprocess.PIPE)
    process.communicate(input=b'\n')

import cv2
import numpy as np

def align_images(rgb_image_path, multispectral_image_path):
    rgb_image = cv2.imread(rgb_image_path)
    multispectral_image = cv2.imread(multispectral_image_path)
    rgb_gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    multispectral_gray = cv2.cvtColor(multispectral_image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(rgb_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(multispectral_gray, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    aligned_rgb = cv2.warpPerspective(rgb_image, M, (multispectral_image.shape[1], multispectral_image.shape[0]))
    return aligned_rgb


def main(input_folder,output_folder):
    ms_photos = find_files(input_folder, [".tif"])
    rgb_photos = find_files(input_folder, [".jpg"])
    sorted_data = sorted(ms_photos, key=sort_key)
    grouped = [list(g) for _, g in groupby(sorted_data, key=lambda x: x.split('DJI_')[-1].split('_')[1])]

    for x in grouped:
        print(x)
        # print()
        pre_process_M3M(x[0],os.path.join(output_folder,'pre_processed_' + x[0].split('\\')[-1]))
        # break
        pre_process_M3M(x[1],os.path.join(output_folder,'pre_processed_' + x[1].split('\\')[-1]))
        pre_process_M3M(x[2],os.path.join(output_folder,'pre_processed_' + x[2].split('\\')[-1]))
        pre_process_M3M(x[3],os.path.join(output_folder,'pre_processed_' + x[3].split('\\')[-1]))
        aligned_rgb_image = align_images(x[0][:-8] + 'D.JPG', 
                                 os.path.join(output_folder,'pre_processed_' + x[0].split('\\')[-1]))
        cv2.imwrite(os.path.join(output_folder,'pre_processed_' + os.path.basename(x[0][:-8] + 'D.JPG')), aligned_rgb_image)
        out_img = os.path.join(output_folder,'pre_processed_' + os.path.basename(x[0][:-8] + 'D.JPG'))
        input_img = x[0][:-8] + 'D.JPG'
        command1 = f'"C:/exiftool(-k).exe" -tagsfromfile "{input_img}" -r -all:all -xmp:all -File:ImageSize "{out_img}"'
        process = subprocess.Popen(command1, shell=True, stdin=subprocess.PIPE)
        process.communicate(input=b'\n')


if __name__ == '__main__':
    input_folder = r"C:\Users\User\Desktop\Fizza\MultiSpectral-Image-Correction\set"
    output_folder = r"C:\Users\User\Desktop\Fizza\MultiSpectral-Image-Correction\outputset"
    main(input_folder,output_folder)


# import os
# import re
# import tifffile
# import subprocess
# def parse_value(text, keyword):
#     pattern = re.escape(keyword) + r'\s*=\s*"([^"]*)"'
#     match = re.search(pattern, text)
#     if match:
#         return match.group(1)
#     else:
#         return None

# def process_tif(file_path, d_file_path):
#     with tifffile.TiffFile(file_path) as tif:
#         metadata = tif.pages[0].tags
#         tif_tags = {}
#         tif_tags['XMP'] = tif.pages[0].tags[700].value.decode('utf-8')

#     altitude = parse_value(tif_tags['XMP'], "drone-dji:AbsoluteAltitude")
#     latitude = parse_value(tif_tags['XMP'], "drone-dji:GpsLatitude")
#     longitude = parse_value(tif_tags['XMP'], "drone-dji:GpsLongitude")
#     pitch = parse_value(tif_tags['XMP'], "drone-dji:FlightPitchDegree")
#     roll = parse_value(tif_tags['XMP'], "drone-dji:FlightRollDegree")
#     yaw = parse_value(tif_tags['XMP'], "drone-dji:FlightYawDegree")
#     command = f'"C:/exiftool(-k).exe" -GpsLatitude="{latitude}" -GpsLongitude="{longitude}" -AbsoluteAltitude="{altitude}" -FlightPitchDegree="{pitch}" -FlightRollDegree="{roll}" -FlightYawDegree="{yaw}" "{d_file_path}"'
    # process = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE)
    # process.communicate(input=b'\n')

# def process_all_tifs(input_folder, out_folder):
#     for root, dirs, files in os.walk(input_folder):

#         for file in files:
#             if file.endswith("_MS_NIR.TIF"):
#                 print(file)
#                 file_path = os.path.join(root, file)
#                 d_file_path = os.path.join(out_folder, file[:-10] + 'D.JPG')
#                 process_tif(file_path, d_file_path)

# input_folder = r"C:\Users\User\Downloads\MS_rawImages"
# out_folder = r"C:\Users\User\Downloads\UpdateRGB"

# process_all_tifs(input_folder, out_folder)