#ifndef __CAR_CONTROL_H__
#define __CAR_CONTROL_H__

#include <Arduino.h>

/*
 *  [0]--|||--[1]
 *   |         |
 *   |         |
 *   |         |
 *   |         |
 *  [3]-------[2]
 */

/** Set the pins for the motors */
#define MOTOR_PINS       (uint8_t[8]){3, 4, 5, 6, A3, A2, A1, A0} 
/** Set the positive and negative directions for the motors */
#define MOTOR_DIRECTIONS (uint8_t[4]){1, 0, 0, 1}

void start_car();
void set_motors(int8_t power0, int8_t power1, int8_t power2, int8_t power3);
void stop_car();
void move(int16_t angle, int8_t power, int8_t rot=0, bool drift=false);

#endif // __CAR_CONTROL_H__

