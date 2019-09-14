from tb_control.dynamixel_interface import Dynamixel
import time
import ipdb
ipdb.set_trace()
d = Dynamixel('/dev/ttyUSB0', 2900)
d.move_to_angle(-60)
cur_pos = d.get_pos()
print((cur_pos - 2900) * 360 / 4096)
#new_pos = d.get_pos() - 25 

