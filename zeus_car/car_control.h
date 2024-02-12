#ifndef __CAR_CONTROL_H__
#define __CAR_CONTROL_H__

#include <Arduino.h>
#include "typedefs.h"


void start_motors();
void stop_motors();
void move(const Action& action);

#endif // __CAR_CONTROL_H__
