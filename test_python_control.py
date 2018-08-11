from testbench_control import TestBench
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

x = 1400
y = 6000
z = 0

mX = 6000
mY = 12000
mZ = 2000

frame_num = 0

tb.target_pos(x, y, 1450)

while tb.busy():
    tb.update()
    cap.grab()

ret, frame = cap.read()
cv2.imwrite("nothing.png", frame)

time.sleep(.5)

tb.reset()

while tb.busy():
    tb.update()




#for i in range(3):
#
#    tb.target_pos(x, y, 0)
#
#    while tb.busy():
#        tb.update()
#        cap.grab()
#    
#    ret, frame = cap.read()
#    cv2.imwrite("nothing.png", frame)
#
#    tb.press_z(800, 10)
#
#    while tb.busy():
#        tb.update()
#        cap.grab()
#
#    time.sleep(.5)
#    data = tb.reqData()
#    ret, frame = cap.read()
#    cv2.imwrite("cap_frame" + str(i) + ".png", frame)
#
#    tb.reset_z()
#
#    while tb.busy():
#        tb.update()
#        cap.grab()
#
#    x += 1300
#
#while tb.busy():
#    tb.update()
#
#tb.reset()
#
#



