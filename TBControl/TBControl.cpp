#include "Arduino.h"
#include "TBControl.h"
#include "Axis.h"
#include "HX711.h"

double TBControl::scaleCalibFactors[] = {-7050.0, -7050.0, -7050.0, -7050.0 };


TBControl::TBControl(Axis *x, Axis *y, Axis *z, HX711 *s) {
    xaxis = x;
    yaxis = y;
    zaxis = z;
    scales = s;
    //init_scales();
}

void TBControl::initialize() {
    xaxis->reset();
    yaxis->reset();
    zaxis->reset();
}

void TBControl::setTarget(int x, int y, int z) {
    xaxis->setTarget(x);
    yaxis->setTarget(y);
    zaxis->setTarget(z);
}

void TBControl::step() {
    xaxis->stepBegin();
    yaxis->stepBegin();
    delayMicroseconds(500);
    xaxis->stepEnd();
    yaxis->stepEnd();
    delayMicroseconds(500);
}

void TBControl::moveZ() {
    zaxis->moveToTargetBlocking();
}

void TBControl::resetZ() {
    zaxis->reset();
}

int TBControl::xPos() {
    return xaxis->getPos();
}

int TBControl::yPos() {
    return yaxis->getPos();
}

int TBControl::zPos() {
    return zaxis->getPos();
}

bool TBControl::xyMoving() {
    return xaxis->moving() || yaxis->moving();
}

void TBControl::tare() {
    for (int i = 0; i < 4; i++) {
        (*(scales + i)).tare();
    }
}

void TBControl::init_scales() {
    for (int i = 0; i < 4; i++) {
        (*(scales + i)).set_scale(scaleCalibFactors[i]);
    }
    tare();
}

void TBControl::log() {
    Serial.print("X: ");
    Serial.print(xPos());
    Serial.print(" Y: ");
    Serial.print(yPos());
    Serial.print(" Z: ");
    Serial.print(zPos());
    for (int i = 0; i < 4; i++) {
        Serial.print(i+1);
        Serial.print(": ");
        Serial.print((*(scales + i)).get_units());
    }
    Serial.println();
}

