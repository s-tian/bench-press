from testbench_control import TestBench
import getch
import time

tb = TestBench('/dev/ttyACM0', 0)

while not tb.ready():
    time.sleep(0.1)
    tb.update()

tb.start()

while tb.busy():
    tb.update()
       
x = 0
y = 0
z = 0

def addx(p):
    global x
    x += p
 
def addy(p):
    global y
    y += p
 
def addz(p):
    global z
    z += p/2

step = 50

inps = {'w': lambda: addx(step), 's': lambda: addx(-step), 'a': lambda: addy(step), 'd': lambda: addy(-step), 'i': lambda: addz(-step), 'k': lambda: addz(step)}

while True:
    ch = getch.getch()
    if ch in inps:
        inps[ch]()
    tb.target_pos(x, y, z)
    tb.update()


