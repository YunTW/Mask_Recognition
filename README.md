# Face Recognition and Classification using Keras and OpenCV

This Python program uses Keras for deep learning and OpenCV for computer vision to perform real-time face detection and classification. It loads a pre-trained deep learning model to recognize faces in webcam video streams and displays the detected faces along with their predicted classes and confidence scores.

## Install

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
