#ifndef Axis_h
#define Axis_h

#include "Arduino.h"

class Axis {
    public:
        Axis(int step, int dir, int limit, bool reverse);
        void reset();
        void setTarget(int target);
        void stepBegin();
        void stepEnd();
        int getPos();
        void moveToTargetBlocking();
        bool moving();
    private:
        void stepBlocking();
        void stepBlocking(int numSteps);
        void setForward();
        void setBackward();
        int stepPin;
        int dirPin;
        int limitPin;
        bool limitState;
        int position;
        int direction;
        int target;
        bool rev;
        int maxSteps;
};


#endif
