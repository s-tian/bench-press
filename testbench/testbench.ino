#include "Arduino.h"
#include "TBControl.h"

const int xStepPin = 40;
const int xDirPin =  38;
const int xLimPin =  24;
const int yStepPin = 36;
const int yDirPin =  34;
const int yLimPin =  22;
const int zStepPin =  0;
const int zDirPin =   0;
const int zLimPin =  26;
const int dOut1 =     0;
const int clk1 =      0;
const int dOut2 =     0;
const int clk2 =      0;
const int dOut3 =     0;
const int clk3 =      0;
const int dOut4 =     0;
const int clk4 =      0;

Axis xAxis(xStepPin, xDirPin, xLimPin, true, 6000);
Axis yAxis(yStepPin, yDirPin, yLimPin, false, 12000);
Axis zAxis(zStepPin, zDirPin, zLimPin, false, 1000);
//
//HX711 scales[4] = {
//    HX711(dOut1, clk1),
//    HX711(dOut2, clk2),
//    HX711(dOut3, clk3),
//    HX711(dOut4, clk4)
//};

HX711 scales [4] = {

  
};
TBControl tb(&xAxis, &yAxis, &zAxis, scales);

void setup() {
    Serial.begin(9600);
    Serial.println("Starting testbench...");
}

String input = "";
int rx_byte;
bool idle = true;

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
        if (tb.xyMoving()) {
            tb.step();
        } else {
            //tb.moveZ();
            Serial.print("Moved to: x:");
            Serial.print(tb.xPos());
            Serial.print(" y:");
            Serial.print(tb.yPos());
            Serial.print(" z:");
            Serial.println(tb.zPos());
            //tb.resetZ();
            idle = true;
        }
    }
}


void handleInput(String s) {
    Serial.println(s);
    if (s == "start") {
        tb.initialize();
        Serial.println("Initialized");
    } else if (s == "r") {
        tb.initialize();
        Serial.println("Reset");
    } else if (s == "l") {
        tb.log();
    } else {
        // By default, a position command
        int xTarget = s.substring(s.indexOf('x') + 1, s.indexOf('y')).toInt();    
        int yTarget = s.substring(s.indexOf('y') + 1, s.indexOf('z')).toInt();
        int zTarget = s.substring(s.indexOf('z') + 1).toInt();
        tb.setTarget(xTarget, yTarget, zTarget);
        idle = false;
    }
}


