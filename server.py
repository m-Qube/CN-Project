# from email import message
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



sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

udp_host = socket.gethostname()
udp_port = 12345


sock.bind((udp_host, udp_port))

while True:
    m, addr = sock.recvfrom(1024)
    m = str(m.decode())
    print(m)
    #print(m)
    print('waiting turn')
    vid = cv2.VideoCapture(-1)
    msg = ''
    while(True):
        print('scanning')
        ret, frame = vid.read()
        cv2.imshow('frame', frame)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            turn = predict(frame)
            vid.release()
            cv2.destroyAllWindows()
            break
    # turn = input('player 2 enter your choice: rock/paper/scissor ')
    print(turn)

    moves = ['rock','paper','scissor']
    if m in moves and turn in moves:
        if m==turn:
            msg = 'tie'
        elif m==moves[0] and turn==moves[1]:
            msg = 'player 2 wins'
        elif m==moves[0] and turn==moves[2]:
            msg = 'player 1 wins'
        elif m==moves[1] and turn==moves[0]:
            msg = 'player 1 wins'
        elif m==moves[1] and turn==moves[2]:
            msg = 'player 2 wins'
        elif m==moves[2] and turn==moves[0]:
            msg = 'player 1 wins'
        elif m==moves[2] and turn==moves[1]:
            msg = 'player 1 wins'
    else:
        msg = 'invalid input'
    
    print(msg)

    sock.sendto(msg.encode(), addr)


