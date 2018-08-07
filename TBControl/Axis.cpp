#include "Arduino.h"
#include "Axis.h"

Axis::Axis(int step, int dir, int limit, bool reverse) {
    stepPin = step;
    dirPin = dir;
    limitPin = limit;
    rev = reverse;
    limitState = 1;
    pinMode(stepPin, OUTPUT);
    pinMode(dirPin, OUTPUT);
    pinMode(limitPin, INPUT);
    setBackward();
}

void Axis::reset() {
    while (limitState || digitalRead(limitPin)) { // Try to prevent false positive
        limitState = digitalRead(limitPin);
        stepBlocking();
    }
    position = 0;
}

void Axis::setTarget(int newTarget) {
    target = newTarget; 
    Serial.print("Target set to ");
    Serial.println(target);
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
    if (rev) {
        digitalWrite(dirPin, LOW);
    } else {
        digitalWrite(dirPin, HIGH);
    }
    Serial.println("set forwa");
    direction = 1;
}

void Axis::setBackward() {
    if (rev) {
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

int Axis::getPos() {
    return position; 
}

bool Axis::moving() {
    return position != target;
}