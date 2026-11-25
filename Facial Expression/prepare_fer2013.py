import os
import csv
import argparse
import numpy as np
import cv2

LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_image(pixels, out_path, size=(48, 48)):
    arr = np.fromstring(pixels, dtype=int, sep=' ')
    arr = arr.reshape(size)
    # FER2013 pixels are grayscale with values 0-255
    cv2.imwrite(out_path, arr)


def main(csv_path, out_dir='data'):
    if not os.path.isfile(csv_path):
        raise SystemExit(f'CSV file not found: {csv_path}')

    counts = {k: 0 for k in LABELS}

    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            emotion = int(row['emotion'])
            pixels = row['pixels']
            usage = row.get('Usage', row.get('usage', 'Training'))

            if usage == 'Training':
                split = 'train'
            elif usage == 'PublicTest':
                split = 'val'
            else:
                split = 'test'

            label_name = LABELS[emotion]
            save_dir = os.path.join(out_dir, split, label_name)
            ensure_dir(save_dir)

            img_name = f'{label_name}_{counts[label_name]:06d}.png'
            out_path = os.path.join(save_dir, img_name)

            save_image(pixels, out_path)
            counts[label_name] += 1

    print('Finished converting FER2013.')
    for k, v in counts.items():
        print(f'{k}: {v}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert FER2013 CSV to folder layout')
    parser.add_argument('--csv', required=True, help='Path to fer2013.csv')
    parser.add_argument('--out', default='data', help='Output folder (default: data)')
    args = parser.parse_args()
    main(args.csv, args.out)
