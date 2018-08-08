from testbench_control import TestBench
from enum import Enum
import time

tb = TestBench('/dev/ttyS4')

while not tb.ready():
    time.sleep(0.1)
    tb.update()

tb.start()

x = 0
y = 0

mX = 6000
mY = 12000

while True:
    tb.update()
    if not tb.busy() and x < 6000 and y < 12000:
        tb.reqData()
        time.sleep(0.5)
        tb.target_pos(x, y, 0)
        x += mX/10
        y += mY/10







