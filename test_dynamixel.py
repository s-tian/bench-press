from tb_control.dynamixel_interface import Dynamixel
import time
import ipdb
ipdb.set_trace()
d = Dynamixel('/dev/ttyUSB0', 2900)
cur_pos = d.get_pos()
print(cur_pos)

new_pos = d.get_pos() - 25 
d.set_pos(new_pos)

