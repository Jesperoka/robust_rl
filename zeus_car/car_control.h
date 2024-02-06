#ifndef __CAR_CONTROL_H__
#define __CAR_CONTROL_H__

#include <Arduino.h>

void start_car();
void set_motors(int8_t omega_0, int8_t omega_1, int8_t omega_2, int8_t omega_3);
void stop_car();
void move(int8_t angle, int8_t power, int8_t rot_vel = 0);

#endif // __CAR_CONTROL_H__
