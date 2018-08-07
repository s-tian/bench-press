#include "Arduino.h"
#include "Axis.h"

TBControl::TBControl(Axis *x, Axis *y, Axis *z) {
    xaxis = x;
    yaxis = y;
    zaxis = z;
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
    return xaxis->position();
}

int TBControl::yPos() {
    return yaxis->position();
}

int TBControl::zPos() {
    return zaxis->position();
}

bool TBControl::xyMoving() {
    return x->moving() || y->moving();
}
