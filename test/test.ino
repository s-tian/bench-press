#include "Arduino.h"
#include "TBControl.h"

#define xStepPin ;
#define xDirPin ;
#define yStepPin ;
#define yDirPin ;
#define zStepPin;
#define zDirPin ;

Axis xAxis(xStepPin, xDirPin);
Axis yAxis(yStepPin, yDirPin);
Axis zAxis(zStepPin, zDirPin);

TBControl tb(&xAxis, &yAxis, &zAxis);

void setup() {
    Serial.begin(9600);
    Serial.println("Starting testbench...");
    tb.initialize();
    Serial.println("Initialized");
}

void loop() {



}

