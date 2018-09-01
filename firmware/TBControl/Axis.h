#ifndef Axis_h
#define Axis_h

#include "Arduino.h"

class Axis {
    public:
        Axis(int step, int dir, int limit, bool reverse, int max);
        void reset();
        void setTarget(int target);
        void stepBegin();
        void stepEnd();
        int getPos();
        void moveToTargetBlocking();
        bool moving();
        void stepBlocking();
        void stepBlocking(int numSteps);
        void setForward();
        int distToTarget();

    private:
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
