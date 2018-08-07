#ifndef Axis_h
#define Axis_h

#include "Arduino.h"

class Axis {
    public:
        Axis(int step, dir);
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
        int position;
        int direction;
}


#endif
