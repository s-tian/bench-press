#include "Arduino.h"
#include "Axis.h"

Axis::Axis(int step, int dir, int limit, bool reverse) {
    stepPin = step;
    dirPin = dir;
    limitPin = limit;
    reverse = reverse;
    limitState = 0;
    pinMode(stepPin, OUTPUT);
    pinMode(dirPin, OUTPUT);
    setBackward();
}

void Axis::reset() {
    while (limitState) {
        limitState = digitalRead(limitPin);
        stepBlocking();
    }
    position = 0;
}

void Axis::setTarget(int newTarget) {
    target = newTarget; 
    if (target > position) {
        setForward();
    } else {
        setBackward();
    }
}

void Axis::stepBlocking(int numSteps) {
    for (int i = 0; i < numSteps; i++) {
        stepBlocking(); 
    }
}

void Axis::moveToTargetBlocking() {
    while (position != target) {
        stepBlocking();
    }
}

void Axis::stepBlocking() {
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(500);
    digitalWrite(stepPin, LOW);
    delayMicroseconds(500);
}

void Axis::setForward() {
    if (reverse) {
        digitalWrite(dirPin, LOW);
    } else {
        digitalWrite(dirPin, HIGH);
    }
    direction = 1;
}

void Axis::setBackward() {
    if (reverse) {
        digitalWrite(dirPin, HIGH);
    } else {
        digitalWrite(dirPin, LOW);
    }
    direction = -1;
}

void Axis::stepBegin() {
    if (position != target) {
        digitalWrite(stepPin, HIGH);
    }
}

void Axis::stepEnd() {
    if (position != target) {
        digitalWrite(stepPin, LOW);
        position += direction;
    }
}

int Axis::position() {
    return position; 
}

bool Axis::moving() {
    return position != target;
}
