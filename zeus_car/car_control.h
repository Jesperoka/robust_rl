#ifndef __CAR_CONTROL_H__
#define __CAR_CONTROL_H__

#include <Arduino.h>


typedef struct {
    float front_left;
    float front_right;
    float back_right;
    float back_left;
} WheelSpeeds;

void start_motors();
void set_motors(WheelSpeeds speeds);
void stop_motors();
void move(uint8_t angle, uint8_t power, uint8_t rot_vel = 0);

#endif // __CAR_CONTROL_H__
