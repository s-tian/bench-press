import serial
import datetime
from enum import Enum

# This class provides an interface to the gelsight testbench setup via serial.
class State(Enum):
    IDLE = 0
    BUSY = 1
    READY = 2

class TestBench():

    IDLE_MSGS = ["Initialized", "Moved", "Reset", "Pressed"]

    def __init__(self, name):
        self.ser = serial.Serial(name, baudrate=9600, timeout=1)
        self.currmsg = ""
        self.state = State.IDLE

    def target_pos(self, x, y, z):
        """ 
        Command testbench to visit an xyz position.
        After calling target_pos, wait for the testbench to become idle again.
        """

        msg = 'x' + str(x) + 'y' + str(y) + 'z' + str(z) + '\n'
        self.ser.write(msg.encode())
        self.state = State.BUSY
        print(self.state)

    def reset(self):
        """
        Command testbench to reset using limit switches and reestablish 
        the origin.
        After calling reset, wait for the testbench to become idle again.
        """

        self.ser.write(b'r\n') 
        self.ser.flush()
        self.state = State.BUSY

    def press_z(self, quick_steps, thresh):
        """
        Command testbench to descend in the z direction in small steps
        until the average threshold force is detected by the load cells.
        See TBControl::feedbackMoveZ, where the actual logic is. This is just
        a layer of serial communication.
        After calling press_z, wait for the testbench to become idle again.
        """
    
        msg = 'pz' + str(quick_steps) + 'w' + str(thresh) + '\n'
        self.ser.write(msg.encode())
        self.ser.flush()
        self.state = State.BUSY

    def reset_z(self):
        """
        Command testbench to reset the Z axis ONLY using the limit switch,
        re-establishing the origin.
        After calling reset_z, wait for the testbench to become idle again. 
        """

        self.ser.write(b'rz\n')
        self.ser.flush()
        self.state = State.BUSY

    def busy(self):
        return self.state == State.BUSY

    def ready(self):
        return self.state == State.READY

    def start(self):
        """
        Command testbench to complete init sequence. 
        This means resetting the axes and re-establishing the origin, as
        well as initializing and tareing the load cells.
        After calling start, wait for the testbench to become idle again.
        """

        self.ser.write(b'start\n')
        self.ser.flush()
        self.state = State.BUSY

    def __handle_msg(self, msg):
        pm = str(datetime.datetime.now()) + ": " + msg
        if any([msg.startswith(key) for key in self.IDLE_MSGS]):
            self.state = State.IDLE
        if msg.startswith("Starting"):
            self.state = State.READY
        print(pm)
        return pm

    def update(self):
        """
        If you are waiting on a message (for example, indicator that state will 
        change from busy to idle), call update in a loop, otherwise new messages
        will not be received over serial.
        """

        for i in range(self.ser.inWaiting()):
            ch = self.ser.read().decode()
            if ch == "\n":
                self.__handle_msg(self.currmsg)
                self.currmsg = ""
            else:
                self.currmsg += ch

    def reqData(self):
        """
        Queries testbench for latest XYZ position and load cell readings. 
        This method, unlike other commands sent to the testbench, 
        is _synchronous__. The state does not change to busy, and the string 
        returned is directly the data string.
        """

        self.ser.write(b'l\n')
        self.ser.flush()
        data = self.ser.readline()
        while data.decode().startswith('l'): # Ignore echo of log request
            data = self.ser.readline()
        return data.decode()


