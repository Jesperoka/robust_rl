#include "car_control.h"

#include <Arduino.h>
#include <SoftPWM.h>

#define MOTOR_POWER_MIN 28
#define MOTOR_POWER_MAX 255
#define MOTOR_START_POWER 100

#define MOTOR_PINS                                                             \
    (uint8_t[8]) { 3, 4, 5, 6, A3, A2, A1, A0 }

/** Set the positive and negative directions for the motors */
#define MOTOR_DIRECTIONS                                                       \
    (uint8_t[4]) { 1, 0, 0, 1 }

int32_t _lastError = 0;
int32_t errorIntegral = 0;
int16_t originHeading;

/** @brief Set speed for 4 motors
 * @param powerX  0 ~ 100 */
void set_motors(int8_t power0, int8_t power1, int8_t power2, int8_t power3) {
    bool dir[4];
    int8_t power[4] = {power0, power1, power2, power3};
    int8_t newPower[4];

    for (uint8_t i = 0; i < 4; i++) {
        dir[i] = power[i] > 0;

        if (MOTOR_DIRECTIONS[i])
            dir[i] = !dir[i];

        if (power[i] == 0) {
            newPower[i] = 0;
        } else {
            newPower[i] = map(abs(power[i]), 0, 100, MOTOR_POWER_MIN, 255);
        }

        if (newPower[i] != 0 && newPower[i] < MOTOR_START_POWER) {
            SoftPWMSet(MOTOR_PINS[i * 2], dir[i] * MOTOR_START_POWER);
            SoftPWMSet(MOTOR_PINS[i * 2 + 1], !dir[i] * MOTOR_START_POWER);
            delayMicroseconds(200);
        }
        SoftPWMSet(MOTOR_PINS[i * 2], dir[i] * newPower[i]);
        SoftPWMSet(MOTOR_PINS[i * 2 + 1], !dir[i] * newPower[i]);
    }
}

/** @param angle the direction you want the car to move
 * @param power 0 ~ 100 (percentage)
 * @param rot the car fixed rotation angle during the movement */
void move(int16_t angle, int8_t power, int8_t rot, bool drift) {
    angle += 90; // 0 is forward
    float rad = angle * PI / 180;

    float ratio = 0.5;

    power /= sqrt(2);
    power = power * (1 - ratio);

    int8_t power_0 = (power * sin(rad) - power * cos(rad)) - rot * ratio;
    int8_t power_1 = (power * sin(rad) + power * cos(rad)) + rot * ratio;
    int8_t power_2 = (power * sin(rad) - power * cos(rad)) + rot * ratio;
    int8_t power_3 = (power * sin(rad) + power * cos(rad)) - rot * ratio;

    set_motors(power_0, power_1, power_2, power_3);
}
