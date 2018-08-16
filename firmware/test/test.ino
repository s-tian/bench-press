#include "Arduino.h"
#include "TBControl.h"

const int xStepPin =40;
const int xDirPin =38;
const int yStepPin =36;
const int yDirPin =34;
const int zStepPin= 32;
const int zDirPin = 30;

Axis xAxis(xStepPin, xDirPin, 24, true);
Axis yAxis(yStepPin, yDirPin, 22, false);
Axis zAxis(zStepPin, zDirPin, 26, false);

TBControl tb(&xAxis, &yAxis, &zAxis);

void setup() {
    Serial.begin(9600);
    Serial.println("Starting testbench...");
    tb.initialize();
    Serial.println("Initialized");

}

void loop() {
}


