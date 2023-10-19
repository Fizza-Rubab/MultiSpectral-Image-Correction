import os
import re
import tifffile
import subprocess
def parse_value(text, keyword):
    pattern = re.escape(keyword) + r'\s*=\s*"([^"]*)"'
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        return None

def process_tif(file_path, d_file_path):
    with tifffile.TiffFile(file_path) as tif:
        metadata = tif.pages[0].tags
        tif_tags = {}
        tif_tags['XMP'] = tif.pages[0].tags[700].value.decode('utf-8')

    altitude = parse_value(tif_tags['XMP'], "drone-dji:AbsoluteAltitude")
    latitude = parse_value(tif_tags['XMP'], "drone-dji:GpsLatitude")
    longitude = parse_value(tif_tags['XMP'], "drone-dji:GpsLongitude")
    pitch = parse_value(tif_tags['XMP'], "drone-dji:FlightPitchDegree")
    roll = parse_value(tif_tags['XMP'], "drone-dji:FlightRollDegree")
    yaw = parse_value(tif_tags['XMP'], "drone-dji:FlightYawDegree")
    command = f'"C:/exiftool(-k).exe" -GpsLatitude="{latitude}" -GpsLongitude="{longitude}" -AbsoluteAltitude="{altitude}" -FlightPitchDegree="{pitch}" -FlightRollDegree="{roll}" -FlightYawDegree="{yaw}" "{d_file_path}"'
    process = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE)
    process.communicate(input=b'\n')

def process_all_tifs(input_folder, out_folder):
    for root, dirs, files in os.walk(input_folder):

        for file in files:
            if file.endswith("_MS_NIR.TIF"):
                print(file)
                file_path = os.path.join(root, file)
                d_file_path = os.path.join(out_folder, file[:-10] + 'D.JPG')
                process_tif(file_path, d_file_path)

input_folder = r"C:\Users\User\Downloads\MS_rawImages"
out_folder = r"C:\Users\User\Downloads\UpdateRGB"

process_all_tifs(input_folder, out_folder)