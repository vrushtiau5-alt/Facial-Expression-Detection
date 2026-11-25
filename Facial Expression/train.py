import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from model import build_emotion_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data', help='Path to data directory with train/val subfolders')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--save', type=str, default='model.h5')
    return parser.parse_args()


def main():
    args = parse_args()

    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')

    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise SystemExit('Data directories not found. Prepare `data/train` and `data/val` with emotion subfolders.')

    train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
    val_datagen = ImageDataGenerator(rescale=1./255)

    target_size = (48, 48)
    color_mode = 'grayscale'

    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=target_size, color_mode=color_mode,
        batch_size=args.batch_size, class_mode='categorical', shuffle=True)

    val_gen = val_datagen.flow_from_directory(
        val_dir, target_size=target_size, color_mode=color_mode,
        batch_size=args.batch_size, class_mode='categorical', shuffle=False)

    num_classes = train_gen.num_classes
    model = build_emotion_model(input_shape=(48, 48, 1), num_classes=num_classes)

    callbacks = [
        ModelCheckpoint(args.save, monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    ]

    model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        callbacks=callbacks
    )


if __name__ == '__main__':
    main()
