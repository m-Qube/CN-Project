from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import cv2

from cvzone.SelfiSegmentationModule import SelfiSegmentation
segmentor = SelfiSegmentation()

model = load_model("models/model10.h5")
#model.summary()

rps = ['paper','rock','scissors']
def predict(frame):
    frame = np.rot90(frame)
    roi = frame[0:280, 0:335]
    frame = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    x = cv2.resize(frame, (100, 100))
    x = x / 255
    img_arr = np.array(x)
    img = np.array([img_arr])
    y = model.predict(img)
    p_num = np.argmax(y, axis=1)[0]
    return rps[p_num]




vid = cv2.VideoCapture(0)

while(True):
    ret, frame = vid.read()
    img1 = frame
    img1 = segmentor.removeBG(img1,img1.shape)
    img1 = cv2.resize(img1,(700,500),None,0.5,0.5)
    cv2.imshow('frame', img1)
    # print(frame.shape)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(predict(frame))
        pass


vid.release()
cv2.destroyAllWindows()