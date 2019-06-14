
# Testbench
This repository contains code for operating the testbench (modified 3 axis CNC). This README will serve as a living document for the rig. 

# Hardware
[Link to CAD files for the hardware setup](https://drive.google.com/drive/u/3/folders/1_D5uNwMah01uUej5cnVeReErW3PcaLc2?usp=drive_open)

The rig is a [2418 "Engraving Machine"](https://www.amazon.com/DIY-Laser-CNC-Kit-Engraving/dp/B01N2510KF/) purchased from Amazon (this link is not quite the one used to purchase the one currently built, because the original link now redirects to a [fancier looking](https://www.amazon.com/Control-Engraving-240x180x45mm-Beauty-Star/dp/B07169D9JQ) one. The dimensions should be verified if purchasing another one of these. 

To begin setup of the bench, there are nice instructions [here](https://imgur.com/gallery/NGafu) which you can follow for the hardware setup.

Once that's set up, there are a few "upgrades" we've made to the rig for our purposes. First of all, limit switches are installed on the X, Y, and Z axes (we call the X axis the one which actually moves the base plate of the rig, the Z axis the one that goes vertically up and down, and Y is the remaining one). The limit switches used are mentioned on the BOM, and for the X and Y axes, are mounted to 3d printed hats which are "friction fit" onto the clamps for the smooth metal rods in each direction. 
*For the X axis, there are two possible mounting locations for the "hat". These are in the "back", closer to the Y axis/electronics, etc, and in the "front". Either location is okay for mounting. We use the "back" mounting location for the ball bearing task, and the "front" for the others, since the bulkier tasks don't allow the CNC to travel to the location without hitting anything.* **If you use the front mounting position, ensure that in the code, you call `tb.flip_x_axis()` before doing anything. This is done in the code for data collection for the dice and analog stick.**
For the Z axis, the screws on the end of the Z axis motor are removed, pushed through a 3d printed part, and into their original place. The Z axis limit switch is then mounted to this 3d printed part. Unfortunately this mounting method doesn't work very well -- it would be good to find a better alternative.

Additionally, four load cells from Sparkfun (again, linked in BOM) have been added underneath the working surface. They are sandwiched by two pieces of aluminum, which have had holes waterjet cut into them for mounting.

Sometimes the axes get a little bit gunky and movements become noticely louder, and I like to apply a bit of lubricant occasionally to keep things moving nicely. I don't quite know which is the best to use at the moment, but I have a small pen of lube that I keep with the testbench stuff. Alternatively, you can get whichever lube you think would be good for this type of thing to use.


# Electronics
The CNC kit comes with a main electronics board which contains a microcontroller, but we instead do most of the work using our own Arduino Mega for control. In order to support all the extra electronics (load cells, limit switches) added on, we also build a custom PCB which should be mounted next to the original electronics board. At this time, the only purpose that the original, CNC electronics board has is **to provide power to the stepper motors**. This is accomplished by wiring out the bottom rows of pins from where the stepper motors used to be, on the original board, to the corresponding pins on the custom board. Wiring all of these pins is *likely not necessary*, and done mostly due to a lack of research done on which pins exactly are needed for power. 

The PCB looks like [this](./doc/pcb.jpg), on the front. Header pins of some kind (male, female) are soldered to all of the through holes, depending on how the external connector works. 

A brief lay of the land: On the left middle-bottom column, are the locations of the three stepper motor drivers.
**Important: The stepper motor drivers get hot during operation. Be careful!**
 In the middle, the largest section is devoted to four boards for reading the load cells. On the top left are pins where the limit switches plug in, and the dense pins in the bottom right connected to the Arduino via a direct physical stacking. In the top left are three optional pins that can be used to wire LEDs for sensors (TODO describe this). *Note that if you are using these pins, you will likely want to solder resistors as shown in the silkscreen, under the top left load cell board.*

The full PCB schematic (Eagle file) is available as a link in the BOM. 

Assuming everything is wired, to run the machine, the Arduino needs to be connected to the computer via USB, and the original CNC electronics board needs to be connected to power via the AC Adapter.

# Software
This repository contains a few things. The first is the firmware code that runs on the Arduino Mega. This is the low level code that describes routines such as moving to a certain position, reading out sensor values, etc. Most likely, this does not need to be modified and will continue to live on the Arduino. If this does need to be changed, program the Arduino through the [Arduino IDE program](https://www.arduino.cc/en/Guide/Linux). 

Next is Python code which interfaces with the Arduino firmware via serial, which can be found in `testbench_control.py`. This code abstracts away all of the serial communication and common functionality, so that you can simply create a Python script, declare a `TestBench` object, say, `tb = TestBench('/dev/ttyACM0', 0)`, where the first two arguments are the serial port which the testbench is connected to and the second is the camera index we would like to read from. The serial port argument is almost always `/dev/ttyACM0` on Linux machines with one Testbench connected. 

## Programming Notes
When using the `TestBench ` class, you should always wait for the Testbench to enter the `ready` state before sending any commands. This can be accomplished by a blocking code block exactly like this:
```
while not tb.ready():
	time.sleep(0.1)
	tb.update()
tb.start()
```
Any time a command is sent to the Testbench which should be completed asynchronously (i.e, any movement command), you should wait in your code until the Testbench is done completing the movement before issuing further commands. For example, after sending a movement command, one might write
```
while tb.busy():
	tb.update()
```
This will allow the testbench to continue receiving sensor and camera data while the motors move. If you do not do this, the camera buffer in particular will fill with frames, and you will experience a strange issue where trying to get frames from the testbench just seems to yield the same frame every time. In reality, they are just adjacent frames in time (and thus look extremely similar), which have not been flushed out of the buffer yet. To fix this, call `tb.update` 5-10 times, which should flush the buffer out before getting a new image. Once `tb.busy() ` evaluates to false, the testbench has completed the task it was instructed to do.

See `test_python_control.py` for an example of how the basic setup of a script might look like (TODO: Update this test script, to not use the camera separately). 

There are pre-written data collection scripts in `data_collection/trajectories/collect_blah_data.py`, which are probably an okay reference for how to set up some data collection with the rig, including doing data logging.

# Before running the robot:
* Make sure that the robot is clear of anything around it 
* Ensure limit switches are mounted -- take extra care on the first reset that it hits the limit switches
* Be ready to kill power at any time (The unfortunate reality). My preferred way of doing this is to disconnected the adapter at the converter (not from the wall or at the DC barrel jack, in the middle)
* If trying to run any script (data collection, etc) overnight, make sure to supervise the **same** script running for ~8 hours before letting it run unsupervised, and it would be good to check with a grad student as well.

Plugging things in:
* Plug in AC adapter from CNC circuit board (DC barrel jack) to wall
* Plug in Arduino to computer through USB
* Plug in any peripherals (dynamixel, etc), which will act as independent units but can be controlled in parallel with the testbench
* Plug in USB cameras (for gelsight, side camera etc)

Please message me (Stephen) with any additional questions.
