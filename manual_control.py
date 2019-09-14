from tb_control.testbench_control import TestBench
import getch
import time

tb = TestBench('/dev/ttyACM0', 0)
#tb.flip_x_reset()

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

step = 200

inps = {'w': lambda: addx(step), 's': lambda: addx(-step), 'a': lambda: addy(step), 'd': lambda: addy(-step), 'i': lambda: addz(-step/2), 'k': lambda: addz(25)}
data = tb.req_data()
x = data['x']
y = data['y']
tb.target_pos(x, y, 0)
x = 0 
y = 0 
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


