from tb_control.dynamixel_interface import Dynamixel
import time
d = Dynamixel('/dev/ttyUSB0')
cur_pos = d.get_pos()
print(cur_pos)

new_pos = d.get_pos() + 1000
d.set_pos(new_pos)

time.sleep(1)
d.set_pos(new_pos - 1000)
