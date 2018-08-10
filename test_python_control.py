from testbench_control import TestBench
from enum import Enum
import time
import cv2

tb = TestBench('/dev/ttyACM0')
cap = cv2.VideoCapture(0)

while not tb.ready():
    time.sleep(0.1)
    tb.update()


tb.start()


while tb.busy():
    tb.update()


tb.reqData()

x = 1700
y = 6000
z = 0

mX = 6000
mY = 12000
mZ = 2000

frame_num = 0


for i in range(3):

    tb.target_pos(x, y, 0)

    while tb.busy():
        tb.update()
        cap.grab()

    tb.press_z(800)

    while tb.busy():
        tb.update()
        cap.grab()

    time.sleep(.5)
    data = tb.reqData()
    ret, frame = cap.read()
    cv2.imwrite("cap_frame" + str(i) + ".png", frame)

    tb.reset_z()

    while tb.busy():
        tb.update()
        cap.grab()

    x += 1000

while tb.busy():
    tb.update()







