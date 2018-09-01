#include "Arduino.h"
#include "Axis.h"

Axis::Axis(int step, int dir, int limit, bool reverse, int max) {
    stepPin = step;
    dirPin = dir;
    limitPin = limit;
    rev = reverse;
    limitState = 1;
    target = 0;
    maxSteps = max;
    pinMode(stepPin, OUTPUT);
    pinMode(dirPin, OUTPUT);
    pinMode(limitPin, INPUT);
}

void Axis::reset() {
    setBackward();
    while (limitState || digitalRead(limitPin)) { // Try to prevent false positive
        limitState = digitalRead(limitPin);
        stepBlocking();
    }
    position = 0;
}

void Axis::setTarget(int newTarget) {
    if (newTarget < 0 || newTarget > maxSteps) {
        Serial.print("Invalid target set at ");
        Serial.println(newTarget);
        return;
    }
    target = newTarget; 
    //Serial.print("Target set to ");
    //Serial.println(target);
    if (target > position) {
        setForward();
    } else {
        setBackward();
    }
}

int Axis::distToTarget() {
    return abs(target - position); 
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
    position += direction;
}

void Axis::setForward() {
    if (rev) {
        digitalWrite(dirPin, LOW);
    } else {
        digitalWrite(dirPin, HIGH);
    }
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
