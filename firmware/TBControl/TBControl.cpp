#include "Arduino.h"
#include "TBControl.h"
#include "Axis.h"
#include "HX711.h"

double TBControl::scaleCalibFactors[] = {-7050.0, -7050.0, -7050.0, -7050.0 };
const int TBControl::FEEDBACK_LIM = 15;

TBControl::TBControl(Axis *x, Axis *y, Axis *z, HX711 *s) {
    xaxis = x;
    yaxis = y;
    zaxis = z;
    scales = s;
}

void TBControl::initialize() {
    zaxis->reset();
    xaxis->reset();
    yaxis->reset();
    init_scales();
}

void TBControl::setTarget(int x, int y, int z) {
    xaxis->setTarget(x);
    yaxis->setTarget(y);
    zaxis->setTarget(z);
}

/**
 * Take a single step in BOTH x and y directions if target not yet reached. 
 * If one axis target has been reached but not the other, move only one.
 */

void TBControl::stepXY() {
    xaxis->stepBegin();
    yaxis->stepBegin();
    delayMicroseconds(500);
    xaxis->stepEnd();
    yaxis->stepEnd();
    delayMicroseconds(500);
}

/**
 * Blocking command to reach target Z position.
 */

void TBControl::moveZToTargetBlocking() {
    zaxis->moveToTargetBlocking();
}

void TBControl::stepZ() {
    zaxis->stepBegin();
    delayMicroseconds(500);
    zaxis->stepEnd();
    delayMicroseconds(500);
}

void TBControl::resetZ() {
    zaxis->reset();
}

/**
 * Move downward in the Z dir until specified avg force reading is reached.
 */

void TBControl::feedbackMoveZ(int fastSteps, double thresh) {
    double weight = 0; 
    int st = 0;
    
    // fastSteps determines how many steps will be 
    // taken quickly, with no load cell feedback.
    zaxis->setForward();
    
    zaxis->stepBlocking(fastSteps);

    while (weight < thresh) {
        zaxis->stepBlocking();
        if (st == 0) {
            weight = 0;
            for (int i = 0; i < 4; i++) {
                weight += abs((*(scales + i)).get_units()); 
            }
        }
        weight /= 4;
        st = (st + 1) % 20;
    }
    Serial.print("Pressed to max force ");
    Serial.println(weight);
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

bool TBControl::zMoving() {
    return zaxis->moving();
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
    delay(500);
    tare();
}

void TBControl::log() {
    Serial.print("X: ");
    Serial.print(xPos());
    Serial.print(" Y: ");
    Serial.print(yPos());
    Serial.print(" Z: ");
    Serial.print(zPos());
    Serial.print(" ");
    for (int i = 0; i < 4; i++) {
        Serial.print(i+1);
        Serial.print(": ");
        Serial.print(abs((*(scales + i)).get_units()));
        Serial.print(" ");
    }
    Serial.println();
}

