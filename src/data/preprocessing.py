import os
import requests
import zipfile
from tqdm import tqdm
import glob
from os import path as osp
import numpy as np

def download_s3dis_dataset(data_dir='./s3dis_data'):
    """Download and extract S3DIS dataset"""
    url = "https://cvg-data.inf.ethz.ch/s3dis/Stanford3dDataset_v1.2_Aligned_Version.zip"

    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "Stanford3dDataset_v1.2_Aligned_Version.zip")
    extract_path = os.path.join(data_dir, "Stanford3dDataset_v1.2_Aligned_Version")

    # Check if already downloaded and extracted
    if os.path.exists(extract_path):
        print("Dataset already exists, skipping download.")
        return extract_path

    print("Downloading S3DIS dataset...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        with open(zip_path, 'wb') as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                pbar.update(size)

        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)

        # Clean up zip file
        os.remove(zip_path)
        print("Dataset download and extraction completed!")

    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please download manually from the provided URL")
        return None

    return extract_path

def preprocess_s3dis(data_dir='./s3dis_data/processed'):
    """Preprocess S3DIS dataset"""
    # Download the dataset
    dataset_path = download_s3dis_dataset()

    # Set up paths and constants
    output_folder = data_dir
    data_dir = dataset_path
    os.makedirs(output_folder, exist_ok=True)

    # S3DIS class names
    class_names = [
        'ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
        'chair', 'table', 'bookcase', 'sofa', 'board', 'clutter'
    ]
    class2label = {one_class: i for i, one_class in enumerate(class_names)}

    # Create annotation paths by scanning dataset
    anno_paths = []
    for area in sorted(os.listdir(data_dir)):
        area_path = os.path.join(data_dir, area)
        if not os.path.isdir(area_path) or not area.startswith('Area_'):
            continue

        for room in sorted(os.listdir(area_path)):
            room_path = os.path.join(area_path, room)
            if not os.path.isdir(room_path):
                continue

            annotations_path = os.path.join(room_path, 'Annotations')
            if os.path.exists(annotations_path):
                anno_paths.append(annotations_path)

    print(f"Found {len(anno_paths)} rooms to process")

    # Fix known issue in Area_5/hallway_6
    revise_file = os.path.join(data_dir, 'Area_5/hallway_6/Annotations/ceiling_1.txt')
    if os.path.exists(revise_file):
        with open(revise_file, 'r') as f:
            data = f.read()
            # replace that extra character with blank space to separate data
            if len(data) > 5545348:
                data = data[:5545347] + ' ' + data[5545348:]

        with open(revise_file, 'w') as f:
            f.write(data)
        print("Fixed Area_5/hallway_6/ceiling_1.txt")

    # Export function
    def export(anno_path, out_filename):
        """Convert original dataset files to points, instance mask and semantic mask files"""
        points_list = []
        ins_idx = 1  # instance ids should be indexed from 1, so 0 is unannotated

        for f in glob.glob(os.path.join(anno_path, '*.txt')):
            one_class = os.path.basename(f).split('_')[0]
            if one_class not in class_names:  # some rooms have 'staris' class
                one_class = 'clutter'

            points = np.loadtxt(f)
            labels = np.ones((points.shape[0], 1)) * class2label[one_class]
            ins_labels = np.ones((points.shape[0], 1)) * ins_idx
            ins_idx += 1
            points_list.append(np.concatenate([points, labels, ins_labels], 1))

        data_label = np.concatenate(points_list, 0)  # [N, 8], (pts, rgb, sem, ins)
        xyz_min = np.amin(data_label, axis=0)[0:3]
        data_label[:, 0:3] -= xyz_min

        np.save(f'{out_filename}_point.npy', data_label[:, :6].astype(np.float32))
        np.save(f'{out_filename}_sem_label.npy', data_label[:, 6].astype(np.int32))
        np.save(f'{out_filename}_ins_label.npy', data_label[:, 7].astype(np.int32))

    # Process all annotation files
    for i, anno_path in enumerate(anno_paths):
        print(f'Processing ({i+1}/{len(anno_paths)}): {anno_path}')

        elements = anno_path.split('/')
        out_filename = elements[-3] + '_' + elements[-2]  # Area_1_hallway_1
        out_filename = os.path.join(output_folder, out_filename)

        if os.path.isfile(f'{out_filename}_point.npy'):
            print('File already exists. skipping.')
            continue

        try:
            export(anno_path, out_filename)
            print(f'Successfully processed: {out_filename}')
        except Exception as e:
            print(f'Error processing {anno_path}: {e}')

    print("Preprocessing completed!")
    print(f"Processed files are saved in: {output_folder}")