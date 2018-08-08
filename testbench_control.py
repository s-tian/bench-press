import serial
import datetime

# This class provides an interface to the gelsight testbench setup via serial.

class TestBench():

    def __init__(self, name):
        self.ser = serial.Serial(name, baudrate=9600)
        self.currmsg = ""

    def target_pos(self, x, y, z):
        msg = 'x' + str(x) + 'y' + str(y) + 'z' + str(z)
        self.ser.write(msg.encode())
        self.waiting = True

    def is_waiting(self):
        return self.waiting

    def start(self):
        self.ser.write(b'start')
        self.waiting = True

    def __handle_msg(self, msg):
        print(str(datetime.datetime.now()) + ": " + msg)
        if msg == "Initialized" or msg.startswith("Moved"):
            self.waiting = False

    # Update must be called in a loop to receive messages!
    def update(self):
        for i in range(self.ser.inWaiting()):
            ch = ser.read().decode()
            if ch == "\n":
                self.__handle_msg(self.currmsg)
                self.currmsg = ""















