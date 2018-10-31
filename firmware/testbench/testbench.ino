#include "Arduino.h"
#include "TBControl.h"

// New PCB Pin mappings

const int xStepPin = 36;
const int xDirPin =  34;
const int xLimPin =  23;
const int yStepPin = 40;
const int yDirPin =  38;
const int yLimPin =  22;
const int zStepPin = 42;
const int zDirPin =  44;
const int zLimPin =  24;
const int dOut1 =    28;
const int clk1 =     26;
const int dOut2 =    30;
const int clk2 =     32;
const int dOut3 =    52;
const int clk3 =     50;
const int dOut4 =    48;
const int clk4 =     46;


/*

Old pin mappings

const int xStepPin = 40;
const int xDirPin =  38;
const int xLimPin =  26;
const int yStepPin = 36;
const int yDirPin =  34;
const int yLimPin =  24;
const int zStepPin = 32;
const int zDirPin =  30;
const int zLimPin =  22;
const int dOut1 =    23;
const int clk1 =     25;
const int dOut2 =    29;
const int clk2 =     27;
const int dOut3 =    53;
const int clk3 =     51;
const int dOut4 =    48;
const int clk4 =     46;

*/

const int xInvInitPos = 8000;

Axis xAxis(xStepPin, xDirPin, xLimPin, true, 8000);
Axis yAxis(yStepPin, yDirPin, yLimPin, false, 12000);
Axis zAxis(zStepPin, zDirPin, zLimPin, true, 1600);

HX711 scales[4] = {
    HX711(dOut1, clk1),
    HX711(dOut2, clk2),
    HX711(dOut3, clk3),
    HX711(dOut4, clk4)
};

TBControl tb(&xAxis, &yAxis, &zAxis, scales);

void setup() {
    Serial.begin(250000);
    Serial.println("Starting testbench...");
}

String input = "";
int rx_byte;
bool idle = true;
double z_force_thresh = 10;
int xInitPos = 0;
int i = 0;

void loop() {
    if (Serial.available()) {
        rx_byte = Serial.read();
        if (rx_byte != '\n') {
            input.concat((char) rx_byte);
        } else {
            handleInput(input);
            input = "";
        }
    }
    if (!idle) {
        while (tb.xyMoving()) {
            tb.stepXY();
        }
       
        //if (tb.zMoving() && (i % 10 != 0 || tb.avgWeight() < z_force_thresh)) {
        if (tb.zMoving()) {
            tb.stepZ();
            //i = (i + 1) % 10;
        } else {
//            Serial.print("Moved to: x:");
//            Serial.print(tb.xPos());
//            Serial.print(" y:");
//            Serial.print(tb.yPos());
//            Serial.print(" z:");
//            Serial.println(tb.zPos());
            Serial.println("Ready");
            
            Serial.flush();
            idle = true;
        }
    }
}

void handleInput(String s) {
    Serial.println(s);
    if (s == "start") {
        tb.initialize(0, 0, xInitPos);
        Serial.println("Initialized");
    } else if (s == "invx") {
        xInitPos = xInvInitPos;
    } else if (s == "r") {
        tb.initialize(0, 0, xInitPos);
        Serial.println("Reset");
    } else if (s == "rz") {
        tb.resetZ();
        Serial.println("Reset Z");
    } else if (s.startsWith("pz")) {
        tb.feedbackMoveZ(s.substring(s.indexOf('z') + 1, s.indexOf('w')).toInt(), s.substring(s.indexOf('w')+1).toInt());
    } else if (s == "l") {
        tb.log();
    } else {
        // By default, a position command
        int xTarget = s.substring(s.indexOf('x') + 1, s.indexOf('y')).toInt();    
        int yTarget = s.substring(s.indexOf('y') + 1, s.indexOf('z')).toInt();
        int zTarget = s.substring(s.indexOf('z') + 1).toInt();
        tb.setTarget(xTarget, yTarget, zTarget);
        tb.moveNorm();
        idle = false;
    }
}
