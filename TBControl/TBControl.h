#ifndef TBControl_h
#define TBControl_h

#include "Arduino.h"
#include "Axis.h"

class TBControl {
    public:
        TBControl(Axis *x, Axis *y, Axis *z); 
        void initialize();
        void setTarget(int x, int y, int z);
        void step();
        void moveZ();
        bool xyMoving();
    private:
        Axis *xaxis;
        Axis *yaxis;
        Axis *zaxis;
}

#endif
