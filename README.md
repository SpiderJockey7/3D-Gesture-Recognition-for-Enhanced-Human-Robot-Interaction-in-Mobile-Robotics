# 3D-Gesture-Recognition-for-Enhanced-Human-Robot-Interaction-in-Mobile-Robotics

This project uses deep learning models to recognise gestures in real-time through a webcam for HCI applications. The system consists of three main components:

1. **Preprocessing**: A script to preprocess your dataset by applying data augmentation.
2. **Training**: A script to train a hand gesture recognition model using ResNet-18.
3. **Recognition**: A script to use the trained model for real-time gesture recognition via a webcam.

## Table of Contents
1. [Preprocessing Dataset](#preprocessing-dataset)
2. [Training the Model](#training-the-model)
3. [Running the Recognition](#running-the-recognition)
4. [Requirements](#requirements)

## Preprocessing Dataset

Before training the model, you need to preprocess your dataset. The dataset should be organized in the following structure:

```
root_folder/
    ├── label_1/
    │    ├── image_1.jpg
    │    ├── image_2.png
    │    └── ...
    ├── label_2/
    │    ├── image_1.jpg
    │    └── ...
    └── ...
```

Each label_x folder represents a different gesture class, and each folder contains images of that gesture. After that, run the preprocessing script: 

```
python preprocess.py <root_folder>
```

The preprocess.py script will apply augmentation (e.g., rotation, brightness, and contrast adjustments) to your images to help improve the model's generalization.

It will process the images and save them in a new folder named **processed_<root_folder>** in the same directory. The images will be resized, augmented, and saved in this folder, ready for training.


## Training the Model
After preprocessing the dataset, you can use the train.py script to train the model on the prepared dataset. If your dataset has more labels, you can set this code's classes to you want:

```
Adjusting for 10 classes
model.fc = nn.Linear(model.fc.in_features, 8) - >  model.fc = nn.Linear(model.fc.in_features, 10)
```

Then run the train.py script:

```
python train.py <processed_folder>
```

Where <processed_folder> is the path to the preprocessed dataset folder.
After training, the model will be saved as the default name **resnet18_model.pth** in the directory where the script is run.


## Running the Recognition
Once the model is trained, you can use the predict.py script to perform real-time hand gesture recognition via a webcam.

You can adjust the number of classes to the number you trained by changing this code:

```
Adjusting for 10 classes
model.fc = nn.Linear(model.fc.in_features, 8) - >  model.fc = nn.Linear(model.fc.in_features, 10)
```

You can change the output of the content you want by changing:

```
Gesture label mapping
gesture_labels = ['CHANGE', 'COMMAND', 'TO', 'ANY', 'OTHER', 'THAT', 'YOU', 'WANT']
```

After that, run the predict.py script:

```
python predict.py <model_file>
```

Features

Gesture Classification: Recognizes 8 different gestures: 'STOP', 'Forward', 'BACK', 'LEFT', 'RIGHT', 'GRAB', 'action_ONE', 'action_TWO'.

Real-time Display: The gesture classification result is displayed on the video stream.
Press ESC to exit the program.

## Requirements

```
Python 3.x
PyTorch
OpenCV
MediaPipe
Pillow
Matplotlib
NumPy
```

Installing the dependencies
You can install the required dependencies using pip:

```
pip install -r requirements.txt
```

Or manually:

```
pip install torch torchvision opencv-python mediapipe pillow matplotlib numpy
```

