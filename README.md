# Face Recognition and Classification using Keras and OpenCV

![Python](https://img.shields.io/badge/Python-14354C.svg?logo=python&logoColor=white) ![OpenCV](https://img.shields.io/badge/Opencv-5C3EE8.svg?logo=Opencv&logoColor=white) ![tensorflow](https://img.shields.io/badge/TensorFlow-FF6F00.svg?logo=tensorflow&logoColor=white)

This Python program uses Keras for deep learning and OpenCV for computer vision to perform real-time face detection and classification. It loads a pre-trained deep learning model to recognize faces in webcam video streams and displays the detected faces along with their predicted classes and confidence scores.

## Table of content

- [Face Recognition and Classification using Keras and OpenCV](#face-recognition-and-classification-using-keras-and-opencv)
  - [Table of content](#table-of-content)
  - [Installation](#installation)
  - [Getting Started](#getting-started)
  - [Usage](#usage)
  - [Contact](#contact)

## Installation

- TensorFlow (required for Keras):

    ```bash
    pip install tensorflow
    ```

- OpenCV: Install it using pip install opencv-python.

    ```bash
    pip install opencv-python
    ```

## Getting Started

Clone the repository or download the source code to your local machine.

Ensure you have the following files in the same directory as the program:

- keras_Model.h5: The pre-trained Keras model for face classification.
- labels.txt: A text file containing class labels corresponding to the model's output classes.
- deploy.prototxt: The prototxt file for face detection using OpenCV DNN.
- res10_300x300_ssd_iter_140000.caffemodel: The pre-trained Caffe model for face detection.

## Usage

Open a terminal or command prompt.

Navigate to the directory where you have the program and the required files.

Run the program using the following command:

```bash
python maskRecognition_cam.py
```

The program will open a webcam window displaying the real-time video feed. It will detect faces in the video and classify them based on the pre-trained model. The recognized class and confidence score will be displayed above each detected face.

To exit the program, press the Esc key (ASCII code 27).

## Contact

[![Github](https://img.shields.io/badge/Github-100000.svg?logo=github&logoColor=white)](https://github.com/YunTW) [![Linkedin](https://img.shields.io/badge/Linkedin-0077B5.svg?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/yuntw/) [![Gmail](https://img.shields.io/badge/Gmail-D14836?logo=gmail&logoColor=white)](terrell60813@gmail.com)
