#include "TBControl.h"

#define xStepPin 3;
#define xDirPin 4;
#define yStepPin 5;
#define yDirPin 6;
#define zStepPin 7;
#define zDirPin 8;

#define xLimPin 0;
#define yLimPin 0;
#define zLimPin 0;

Axis xAxis(xStepPin, xDirPin, xLimPin, false);
Axis yAxis(yStepPin, yDirPin, yLimPin, true);
Axis zAxis(zStepPin, zDirPin, zLimPin, false);

TBControl tb(&xAxis, &yAxis, &zAxis);

void setup() {
    Serial.begin(9600);
    Serial.println("Starting testbench...");
    tb.initialize();
    Serial.println("Initialized");
}

String input = "";

void loop() {
    if (Serial.available()) {
        rx_byte = Serial.read();
        if (rx_byte != '\n') {
            input += rx_byte;
        } else {
            handleInput(input);
            input = "";
        }
    }

    if (tb.xyMoving()) {
        tb.step();
    } else {
        tb.moveZ();
        Serial.print("Moved to: x:");
        Serial.print(xaxis->position());
        Serial.print(" y:");
        Serial.print(yaxis->position());
        Serial.print(" z:");
        Serial.println(zaxis->position());
        tb.resetZ();
    }
}


void handleInput(String s) {
    int xTarget = s.substring(s.indexOf('x') + 1, s.indexOf('y')).toInt();    
    int yTarget = s.substring(s.indexOf('y') + 1, s.indexOf('z')).toInt();
    int zTarget = s.substring(s.indexOf('z') + 1).toInt();
    tb.setTarget(xTarget, yTarget, zTarget);
}


