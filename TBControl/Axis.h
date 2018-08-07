#ifndef Axis_h
#define Axis_h

#include "Arduino.h"

class Axis {
    public:
        Axis(int step, int dir, int limit, bool reverse);
        void reset();
        void setTarget();
        void stepBegin();
        void stepEnd();
        int position();
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
        int position;
        int direction;
        bool reverse;
}


#endif
