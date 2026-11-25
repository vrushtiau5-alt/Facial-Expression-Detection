# Facial Expression Detection

This repository contains a simple real-time facial emotion detection project using a CNN built with Keras (TensorFlow backend) and OpenCV for webcam capture and face detection.

Required Python version: 3.13.0

Recommend using a virtual environment with Python 3.13.0 to ensure package compatibility.

**What you get**
 - `model.py` — CNN model definition (build_emotion_model)
 - `train.py` — training script using `ImageDataGenerator` (expects `data/train` and `data/val` folders)
 - `camera.py` — real-time webcam demo that loads `model.h5` and displays predicted emotion
 - `utils.py` — preprocessing helpers and emotion label list
 - `requirements.txt` — Python dependencies

**Quick start**

1. Create a Python virtual environment and install dependencies:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
pip install -r requirements.txt
```

2. Prepare dataset

- Prepare training and validation folders with subfolders per emotion label (e.g. `Angry`, `Happy`, etc.). Each subfolder should contain grayscale or color face images sized to be resized to `48x48`.
- You can use the FER2013 dataset and convert it into a directory structure. Many tutorials describe converting the FER2013 CSV to folders.

Directory layout expected by `train.py`:

```
data/
	train/
		Angry/
		Disgust/
		Fear/
		Happy/
		Sad/
		Surprise/
		Neutral/
	val/
		Angry/
		...
```

3. Train a model

```bash
python train.py --data_dir data --epochs 50 --batch_size 64 --save model.h5
```

4. Run real-time demo

Make sure `model.h5` exists (from training or a pre-trained model). Then run:

```bash
python camera.py
```

Press `q` to quit the webcam window.

Notes
- `camera.py` uses OpenCV's Haar cascade from `cv2.data.haarcascades` to detect faces.
- The model expects `48x48` grayscale faces normalized to [0,1].
- If you want to use a different dataset layout, modify `train.py` accordingly.

If you want, I can:
- Add a small data preparation helper to convert FER2013 CSV to the directory layout.
- Add a sample pretrained `model.h5` (if you provide one or allow download from a URL).

FER2013 conversion helper
 - `prepare_fer2013.py` — convert the FER2013 `fer2013.csv` file into `data/train`, `data/val`, and `data/test` folders with emotion subfolders. Usage:

```bash
python prepare_fer2013.py --csv path/to/fer2013.csv --out data
```

This produces the layout expected by `train.py` (training images under `data/train/<Emotion>` and validation under `data/val/<Emotion>`).
# Facial-Expression-Detection