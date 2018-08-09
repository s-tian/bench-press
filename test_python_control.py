from testbench_control import TestBench
from enum import Enum
import time

tb = TestBench('/dev/ttyS4')

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

tb.target_pos(3000, 6000, 0)

while tb.busy():
    tb.update()

tb.press_z()

while tb.busy():
    tb.update()

print("not qwef")

tb.reset()

while tb.busy():
    tb.update()






