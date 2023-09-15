from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
import opencv_dnns as dnns

# Disable scientific notation for clarity
# 用於設置 Numpy 對數據的輸出方式。
# 「suppress」參數是一個布爾值，設置為「True」時會強制輸出數據沒有科學記號，也就是小數點後只有一位。
np.set_printoptions(suppress=True)

# Load the model
# compile=false 意味著在載入模型時不需要再次編譯
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Face Detection 資料
prototxt = "deploy.prototxt"
caffemodel = "res10_300x300_ssd_iter_140000.caffemodel"
# 初始化模型 (模型使用的Input Size為 (300, 300))
net = cv2.dnn.readNetFromCaffe(prototxt=prototxt, caffeModel=caffemodel)

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)
if not camera.isOpened:
    print('相機開啟失敗!')
    exit()

while True:
    # Grab the webcamera's image.
    ret, frame = camera.read()

    # Face Detection
    rects = dnns.faceDetect(frame)
    # loop所有預測結果
    for rect in rects:
        (x, y, w, h) = rect["box"]
        confidence = rect["confidence"]

        # 擷取臉部
        face = frame[y:y+h, x:x+w]

        # 畫出邊界框
        cv2.rectangle(frame, (x, y), (x + w, y + h),
                      (0, 255, 0), 2)

        # # 畫出準確率
        # text = f"{round(confidence * 100, 2)}%"
        # y = y - 10 if y - 10 > 10 else y + 10
        # cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
        #             0.45, (0, 0, 255), 2)

        # Resize the raw image into (224-height,224-width) pixels
        image = cv2.resize(face, (224, 224), interpolation=cv2.INTER_AREA)

        # Make the image a numpy array and reshape it to the models input shape.
        # 將圖像數據轉換為陣列，並使用「np.asarray」函數將數據轉換為浮點數陣列，
        # 並使用「reshape」函數將其轉換為一個長度為「1」，寬度為「224」，高度為「224」，通道數為「3」的 4 維數組。
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

        # Normalize the image array
        image = (image / 127.5) - 1

        # Predicts the model
        prediction = model.predict(image)
        index = np.argmax(prediction)  # 求出數組中最大值得索引
        class_name = class_names[index]
        # 「prediction」是一個二維數組，並且我們只需要第一個（和唯一的）預測結果。
        confidence_score = prediction[0][index]

        # # Print prediction and confidence score
        # print("Class:", class_name[2:], end="")
        # print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

        # 畫出分類
        if confidence_score > 0.8:
            text = 'Class: ' + class_name[2:] + ' Score: ' + \
                str(np.round(confidence_score * 100))[:-2] + '%'
        # else:
        #     text = 'Class: Unknow'

        y = y - 10 if y - 10 > 10 else y + 10
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 255), 2)

    # Show face ROI
    # cv2.imshow('face', face)
    # Show the image in a window
    cv2.imshow("Webcam Image", frame)

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
