#ifndef TBControl_h
#define TBControl_h

#include "Arduino.h"
#include "HX711.h"
#include "Axis.h"

class TBControl {
    public:
        TBControl(Axis *x, Axis *y, Axis *z, HX711 *scales); 
        void initialize();
        void setTarget(int x, int y, int z);
        void stepXY();
        void stepZ();
        void moveZToTargetBlocking();
        void resetZ();
        void feedbackMoveZ(int fastSteps, double thresh);
        bool xyMoving();
        bool zMoving();
        int xPos();
        int yPos();
        int zPos();
        void log();
        static double scaleCalibFactors[];
        static const int FEEDBACK_LIM;

    private:
        Axis *xaxis;
        Axis *yaxis;
        Axis *zaxis;
        HX711 *scales;
        void tare();
        void init_scales();
};

#endif
