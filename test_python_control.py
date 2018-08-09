from testbench_control import TestBench
from enum import Enum
import time
import cv2

tb = TestBench('/dev/ttyS4')
cap = cv2.VideoCapture(0)

while not tb.ready():
    time.sleep(0.1)
    tb.update()

tb.start()

while tb.busy():
    tb.update()

x = 0
y = 0
z = 0

mX = 6000
mY = 12000
mZ = 2000

frame_num = 0

tb.target_pos(3800, 5600, 0)

while tb.busy():
    tb.update()

tb.press_z()

while tb.busy():
    tb.update()

data = tb.reqData()
ret, frame = cap.read()
cv2.imwrite("cap_frame" + str(frame_num) + ".jpg", frame)

tb.reset()

while tb.busy():
    tb.update()






