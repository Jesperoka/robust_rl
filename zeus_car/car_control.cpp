#include <Arduino.h>
#include <SoftPWM.h>

#include "car_control.h"


static const uint8_t MOTOR_POWER_MIN = 28; // NOTE: set to 73 for 20% min power // NOTE: why can't it be 0? is it an elevated zero?
static const uint8_t MOTOR_POWER_MAX = 255; // NOTE: set to 210 for 80% max power
static const uint8_t MOTOR_START_POWER = 100; // NOTE: should this be (MOTOR_POWER_MAX - MOTOR_POWER_MIN) / 2? i.e. 113.5
static const uint8_t OMEGA_MIN = -100; // TODO: replace with the actual minimum possible omega
static const uint8_t OMEGA_MAX = 100; // TODO: replace with the actual maximum possible omega


//  Motor layout
//
//  [0]--|||--[1]
//   |         |
//   |         |
//   |         |
//   |         |
//  [3]-------[2]

static const uint8_t[2] MOTOR_0_PIN = {3, 4};
static const uint8_t[2] MOTOR_1_PIN = {5, 6};
static const uint8_t[2] MOTOR_2_PIN = {A3, A2};
static const uint8_t[2] MOTOR_3_PIN = {A1, A0};


void start_car() {
    SoftPWMSet(MOTOR_0_PIN[0], 0);
    SoftPWMSet(MOTOR_0_PIN[1], 0);
    SoftPWMSetFadeTime(MOTOR_0_PIN[0], 100, 100);
    SoftPWMSetFadeTime(MOTOR_0_PIN[1], 100, 100);

    SoftPWMSet(MOTOR_1_PIN[0], 0);
    SoftPWMSet(MOTOR_1_PIN[1], 0);
    SoftPWMSetFadeTime(MOTOR_1_PIN[0], 100, 100);
    SoftPWMSetFadeTime(MOTOR_1_PIN[1], 100, 100);

    SoftPWMSet(MOTOR_2_PIN[0], 0);
    SoftPWMSet(MOTOR_2_PIN[1], 0);
    SoftPWMSetFadeTime(MOTOR_2_PIN[0], 100, 100);
    SoftPWMSetFadeTime(MOTOR_2_PIN[1], 100, 100);

    SoftPWMSet(MOTOR_3_PIN[0], 0);
    SoftPWMSet(MOTOR_3_PIN[1], 0);
    SoftPWMSetFadeTime(MOTOR_3_PIN[0], 100, 100);
    SoftPWMSetFadeTime(MOTOR_3_PIN[1], 100, 100);
}

void stop_car() {
    SoftPWMSet(MOTOR_0_PIN[0], 0);
    SoftPWMSet(MOTOR_0_PIN[1], 0);
    SoftPWMSet(MOTOR_1_PIN[0], 0);
    SoftPWMSet(MOTOR_1_PIN[1], 0);
    SoftPWMSet(MOTOR_2_PIN[0], 0);
    SoftPWMSet(MOTOR_2_PIN[1], 0);
    SoftPWMSet(MOTOR_3_PIN[0], 0);
    SoftPWMSet(MOTOR_3_PIN[1], 0);
}


void move(int8_t angle, int8_t power, int8_t rot_vel = 0) {
    int16_t[4]* omega = mecanum_inverse_kinematics(angle, power, rot_vel);
    set_motors(omega[0], omega[1], omega[2], omega[3]);
}


// https://journals.sagepub.com/doi/10.1177/02783649241228607
// alpha_1,4 = 135 deg
// alpha_2,3 = 45 deg
// gamma_1,4 = alpha_1,4 - 90 deg = 45 deg
// gamma_2,3 = 90 deg  - alpha_2,3 = 45 deg
// R = wheel radius = 0.03 m
// L = half length of wheel front-back distance (from wheel center) = TODO: measure properly
// W = half length of wheel left-right distance (from wheel inner edge) = TODO: measure properly
// W_w = wheel width = TODO: measure properly
// W_t = W + W_w = TODO: measure properly


int16_t[4]* mecanum_inverse_kinematics(int8_t angle, int8_t magnitude, int8_t rot_vel) {
    // TODO: precompute after measuring W_t and L (and R for good measure).
    KI_11 = (2.0/R)*0.5;                // = 1/R
    KI_12 = (2.0/R)*0.5;                // = 1/R
    KI_13 = (2.0/R)*(-0.5)*(W_t + L);   // = -1/R*(W_t + L)
    K1_21 = (2.0/R)*0.5;                // = 1/R
    K1_22 = (2.0/R)*(-0.5);             // = -1/R
    K1_23 = (2.0/R)*0.5*(W_t + L);      // = (W_t + L)/R
    KI_31 = (2.0/R)*0.5;                // = 1/R
    KI_32 = (2.0/R)*(-0.5);             // = -1/R
    KI_33 = (2.0/R)*(-0.5)*(W_t + L);   // = -1/R*(W_t + L)
    KI_41 = (2.0/R)*0.5;                // = 1/R
    KI_42 = (2.0/R)*0.5;                // = 1/R
    KI_43 = (2.0/R)*0.5*(W_t + L);      // = (W_t + L)/R

    // Get precomputed values
    cos_angle = valid_cosines[angle];
    sin_angle = valid_sines[angle];
    rotation = valid_rotations[rot_vel];

    // TODO: precompute the maxium omega value of any given wheel.

    // Calculate the wheel speeds
    omega_0 = magnitude * (KI_11 * cos_angle + KI_12 * sin_angle) + KI_13 * rotation;
    omega_1 = magnitude * (KI_21 * cos_angle + KI_22 * sin_angle) + KI_23 * rotation;
    omega_2 = magnitude * (KI_31 * cos_angle + KI_32 * sin_angle) + KI_33 * rotation;
    omega_3 = magnitude * (KI_41 * cos_angle + KI_42 * sin_angle) + KI_43 * rotation;

    return {omega_0, omega_1, omega_2, omega_3};
}

// Sets the pin PWM values for the motors based on the wheel speeds
void set_motors(int8_t omega_0, int8_t omega_1, int8_t omega_2, int8_t omega_3) {
    // Map wheel speeds to motor pin values
    int8_t front_left_power = map(omega_0, OMEGA_MIN, OMEGA_MAX, MOTOR_POWER_MIN, MOTOR_POWER_MAX);
    int8_t front_right_power = map(omega_1, OMEGA_MIN, OMEGA_MAX, MOTOR_POWER_MIN, MOTOR_POWER_MAX);
    int8_t back_right_power = map(omega_2, OMEGA_MIN, OMEGA_MAX, MOTOR_POWER_MIN, MOTOR_POWER_MAX);
    int8_t back_left_power = map(omega_3, OMEGA_MIN, OMEGA_MAX, MOTOR_POWER_MIN, MOTOR_POWER_MAX);

    bool positive = omega_0 > 0;
    SoftPWMSet(MOTOR_0_PIN[0], !positive*front_left_power);
    SoftPWMSet(MOTOR_0_PIN[1], positive*front_left_power);

    bool positive = omega_3 > 0;
    SoftPWMSet(MOTOR_3_PIN[0], !positive*back_left_power);
    SoftPWMSet(MOTOR_3_PIN[1], positive*back_left_power);

    bool positive = omega_1 > 0;
    SoftPWMSet(MOTOR_1_PIN[0], positive*front_right_power);
    SoftPWMSet(MOTOR_1_PIN[1], !positive*front_right_power);

    bool positive = omega_2 > 0;
    SoftPWMSet(MOTOR_2_PIN[0], positive*back_right_power);
    SoftPWMSet(MOTOR_2_PIN[1], !positive*back_right_power);
}
