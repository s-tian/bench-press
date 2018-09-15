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
       
def addx(p):
    global x
    x += p
 
def addy(p):
    global y
    y += p
 
def addz(p):
    global z
    z += p

step = 50

inps = {'w': lambda: addx(step), 's': lambda: addx(-step), 'a': lambda: addy(step), 'd': lambda: addy(-step), 'i': lambda: addz(-step/5), 'k': lambda: addz(step/5)}

tb.target_pos(2000, 6000, 0)
x = 2000 
y = 6000 
z = 0


while tb.busy():
    tb.update()

while True:
    ch = getch.getch()
    if ch in inps:
        inps[ch]()

    print(tb.req_data())
    tb.target_pos(x, y, z)
    while tb.busy():
        tb.update()


