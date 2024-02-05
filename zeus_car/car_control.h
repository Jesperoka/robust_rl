#ifndef __CAR_CONTROL_H__
#define __CAR_CONTROL_H__

#include <Arduino.h>


void start_car();
void set_motors(int8_t power0, int8_t power1, int8_t power2, int8_t power3);
void stop_car();
void move(int16_t angle, int8_t power, int8_t rot = 0, bool drift = false);

#endif // __CAR_CONTROL_H__
