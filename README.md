# Facial Expression Detection

## 1. Project Overview

Facial Expression Detection is a real-time facial emotion recognition system developed using Deep Learning and Computer Vision techniques.
The system uses a Convolutional Neural Network (CNN) built with TensorFlow and Keras to recognize human facial expressions. OpenCV is used to capture live video through a webcam, detect faces, and display the predicted emotions.
The system identifies seven basic facial expressions:
- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

## 2. Objectives

- To detect human faces using a webcam.
- To recognize facial expressions using a CNN model.
- To implement real-time emotion detection using OpenCV.
- To classify facial expressions into seven emotion categories.
- To demonstrate the application of Artificial Intelligence and Deep Learning in facial emotion recognition.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.13.0 | Programming language |
| TensorFlow | Deep learning framework |
| Keras | CNN model development |
| OpenCV 4.10.0.84 | Face detection and webcam capture |
| NumPy | Numerical operations |
| FER2013 | Facial expression dataset |
| Visual Studio Code | Development environment |

## 4. Project Structure

```text
Facial Expression Detection/
│
├── .git/
├── venv/
│
└── Facial Expression/
    │
    ├── data/
    │   ├── train/
    │   ├── val/
    │   └── test/
    │
    ├── screenshots/
    │   └── facial-expression-output.png
    │
    ├── camera.py
    ├── model.py
    ├── model.h5
    ├── train.py
    ├── util.py
    ├── prepare_fer2013.py
    ├── requirements.txt
    └── README.md

```

## 5. Requirements

- Python 3.13.0
- Visual Studio Code
- Webcam
- Required Python libraries
- FER2013 dataset (for training the model)

## 6. Installation and Setup

### Step 1: Create a Virtual Environment

Open the terminal in the main project directory:

```powershell
python -m venv venv
```

### Step 2: Activate the Virtual Environment

For Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 3: Navigate to the Project Folder

```powershell
cd "Facial Expression"
```

### Step 4: Upgrade pip

```powershell
python -m pip install --upgrade pip
```

### Step 5: Install Dependencies

```powershell
pip install -r requirements.txt
```

Install the compatible OpenCV version:

```powershell
pip install opencv-python==4.10.0.84
```

### Step 6: Verify OpenCV Installation

```powershell
python -c "import cv2; print(cv2.__version__); print(hasattr(cv2, 'CascadeClassifier'))"
```

Expected output:

```text
4.10.0
True
```

## 7. Dataset Preparation

The project uses the FER2013 facial expression dataset.

The dataset contains facial images belonging to different emotion categories.

The expected directory structure is:

```text
data/
│
├── train/
│   ├── Angry/
│   ├── Disgust/
│   ├── Fear/
│   ├── Happy/
│   ├── Sad/
│   ├── Surprise/
│   └── Neutral/
│
├── val/
│   ├── Angry/
│   ├── Disgust/
│   ├── Fear/
│   ├── Happy/
│   ├── Sad/
│   ├── Surprise/
│   └── Neutral/
│
└── test/
```

To convert the FER2013 CSV dataset into the required folder structure, execute:

```powershell
python prepare_fer2013.py --csv path/to/fer2013.csv --out data
```

## 8. Model Training

The project uses a Convolutional Neural Network (CNN) to extract facial features and classify facial expressions.

To train the model, execute:

```powershell
python train.py --data_dir data --epochs 50 --batch_size 64 --save model.h5
```

After training, the model is saved as:

```text
model.h5
```

If a compatible trained model already exists, you can directly use it for real-time detection without training again.

## 9. Real-Time Facial Expression Detection

The `camera.py` file is responsible for real-time facial expression recognition.

It performs the following operations:

1. Captures live video using the webcam.
2. Detects faces using OpenCV Haar Cascade.
3. Extracts the detected facial region.
4. Preprocesses the face image into 48 × 48 grayscale format.
5. Normalizes pixel values between 0 and 1.
6. Uses the trained CNN model to predict the facial expression.
7. Displays the predicted emotion on the webcam window.

### Run the Application

Execute the following command:

```powershell
python camera.py
```

The webcam will open and display the detected facial expression in real time.

Press **Q** to close the webcam window.

## 10. Project Output

The Facial Expression Detection system successfully detects human faces through a webcam and predicts facial emotions in real time using a trained Convolutional Neural Network (CNN) model.

The following screenshots demonstrate the actual output of the project.

### 1. Sad Expression

![Sad Expression](screenshots/sad.png)

The system detects the face and predicts the emotion as **Sad**.

### 2. Happy Expression

![Happy Expression](screenshots/happy.png)

The system detects the face and predicts the emotion as **Happy**.

### 3. Angry Expression

![Angry Expression](screenshots/angry.png)

The system detects the face and predicts the emotion as **Angry**.

### 4. Surprise Expression

![Surprise Expression](screenshots/surprise.png)

The system detects the face and predicts the emotion as **Surprise**.

### Output Summary

The system successfully demonstrates real-time facial expression recognition through webcam input. It identifies faces using OpenCV Haar Cascade and classifies the detected facial expressions using a trained CNN model.
The predicted emotion is displayed above the detected face with a green rectangular bounding box.

**Detected expressions demonstrated:**
- Sad
- Happy
- Angry
- Surprise

## 11. Conclusion

The Facial Expression Detection project demonstrates real-time facial emotion recognition using Artificial Intelligence, Deep Learning, and Computer Vision.
By combining CNN, TensorFlow, Keras, and OpenCV, the system detects human faces and classifies their facial expressions through live webcam input.
