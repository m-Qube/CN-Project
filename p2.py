import socket
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import cv2


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





# while(True):
#     ret, frame = vid.read()
#     cv2.imshow('frame', frame)
#     print(frame.shape)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         print(predict(frame))
#         break

sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)

udp_host = socket.gethostname()
udp_port = 12345

#msg = 'hello'
play = True

while play:
    # msg = input('player 1 enter your choice: rock/paper/scissors ')
    print('scanning')
    vid = cv2.VideoCapture(0)
    while(True):
        ret, frame = vid.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            msg = predict(frame)
            vid.release()
            cv2.destroyAllWindows()
            break
    # msg = ''
    print(msg)
    sock.sendto(msg.encode(),(udp_host,udp_port))

    encodedModified, serverAddress = sock.recvfrom(1024)
    print(encodedModified.decode())

    play = bool(int(input('would you like to play again? (0/1)')))

sock.close()
