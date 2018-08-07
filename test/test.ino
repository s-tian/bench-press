#include "Arduino.h"
#include "TBControl.h"

const int xStepPin =40;
const int xDirPin =38;
const int yStepPin =36;
const int yDirPin =34;
const int zStepPin= 0;
const int zDirPin =0;

Axis xAxis(xStepPin, xDirPin, 24, true);
Axis yAxis(yStepPin, yDirPin, 22, false);
Axis zAxis(zStepPin, zDirPin, 26, false);

TBControl tb(&xAxis, &yAxis, &zAxis);

void setup() {
    Serial.begin(9600);
    Serial.println("Starting testbench...");
    tb.initialize();
    Serial.println("Initialized");
    tb.setTarget(3000, 3000, 0);

}

void loop() {
    tb.step();
}


