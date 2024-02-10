#include <Arduino.h>
#include <SoftPWM.h>

#include "car_control.h"
#include "car_constants.h"


constexpr uint8_t MOTOR_POWER_MIN = 0; // NOTE: set to 73 for 20% min power // NOTE: why can't it be 0? is it an elevated zero?
constexpr uint8_t MOTOR_POWER_MAX = 255; // NOTE: set to 210 for 80% max power
constexpr uint8_t MOTOR_START_POWER = 100; // NOTE: should this be (MOTOR_POWER_MAX - MOTOR_POWER_MIN) / 2? i.e. 113.5
constexpr uint8_t OMEGA_MIN = -21; 
constexpr uint8_t OMEGA_MAX = 21;

//  Motor layout
//
//  [0]--|||--[1]
//   |         |
//   |         |
//   |         |
//   |         |
//  [3]-------[2]
constexpr int8_t MOTOR_0_PIN[2]  = {3, 4};
constexpr int8_t MOTOR_1_PIN[2]  = {5, 6};
constexpr int8_t MOTOR_2_PIN[2]  = {A3, A2};
constexpr int8_t MOTOR_3_PIN[2]  = {A1, A0};

constexpr uint8_t T_FADE = 100;

#define CLAMP(val, limit) (((val) > (limit)) ? (limit) : (val))

void start_motors() {
    SoftPWMSet(MOTOR_0_PIN[0], 0);
    SoftPWMSet(MOTOR_0_PIN[1], 0);
    SoftPWMSetFadeTime(MOTOR_0_PIN[0], T_FADE, T_FADE);
    SoftPWMSetFadeTime(MOTOR_0_PIN[1], T_FADE, T_FADE);

    SoftPWMSet(MOTOR_1_PIN[0], 0);
    SoftPWMSet(MOTOR_1_PIN[1], 0);
    SoftPWMSetFadeTime(MOTOR_1_PIN[0], T_FADE, T_FADE);
    SoftPWMSetFadeTime(MOTOR_1_PIN[1], T_FADE, T_FADE);

    SoftPWMSet(MOTOR_2_PIN[0], 0);
    SoftPWMSet(MOTOR_2_PIN[1], 0);
    SoftPWMSetFadeTime(MOTOR_2_PIN[0], T_FADE, T_FADE);
    SoftPWMSetFadeTime(MOTOR_2_PIN[1], T_FADE, T_FADE);

    SoftPWMSet(MOTOR_3_PIN[0], 0);
    SoftPWMSet(MOTOR_3_PIN[1], 0);
    SoftPWMSetFadeTime(MOTOR_3_PIN[0], T_FADE, T_FADE);
    SoftPWMSetFadeTime(MOTOR_3_PIN[1], T_FADE, T_FADE);
}

void stop_motors() {
    SoftPWMSet(MOTOR_0_PIN[0], 0);
    SoftPWMSet(MOTOR_0_PIN[1], 0);

    SoftPWMSet(MOTOR_1_PIN[0], 0);
    SoftPWMSet(MOTOR_1_PIN[1], 0);
    
    SoftPWMSet(MOTOR_2_PIN[0], 0);
    SoftPWMSet(MOTOR_2_PIN[1], 0);

    SoftPWMSet(MOTOR_3_PIN[0], 0);
    SoftPWMSet(MOTOR_3_PIN[1], 0);
}


// https://journals.sagepub.com/doi/10.1177/02783649241228607 
// Arguments are indices to lookup tables
WheelSpeeds mecanum_inverse_kinematics(uint8_t angle, uint8_t velocity, int8_t rot_vel) {
    angle = CLAMP(angle, ANGLE_DISCRETIZATION - 1);
    velocity = CLAMP(velocity, MAGNITUDE_DISCRETIZATION - 1);
    rot_vel = CLAMP(rot_vel, ROTATIONAL_VELOCITY_DISCRETIZATION - 1);

    const float v = VALID_MAGNITUDES[velocity];
    const float sin_phi = VALID_SINES[angle];
    const float cos_phi = VALID_COSINES[angle];
    const float omega_z = VALID_ROTATIONAL_VELOCITIES[rot_vel];

    const WheelSpeeds wheel_speeds = {
        front_left:     v * (KI_11 * cos_phi + KI_12 * sin_phi) + KI_13 * omega_z,
        front_right:    v * (KI_21 * cos_phi + KI_22 * sin_phi) + KI_23 * omega_z,
        back_right:     v * (KI_31 * cos_phi + KI_32 * sin_phi) + KI_33 * omega_z,
        back_left:      v * (KI_41 * cos_phi + KI_42 * sin_phi) + KI_43 * omega_z,
    };

    return wheel_speeds;
}

// Use with care, function expects output limits to be within uint8_t range
const uint8_t _map(const float x, const float in_min, const float in_max, const float out_min, const float out_max) {
    return (uint8_t)((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min - 1e-6);
}


void overcome_static_friction(bool positive_fl, bool positive_bl, bool positive_fr, bool positive_br) {
    SoftPWMSet(MOTOR_0_PIN[0], (not positive_fl)*MOTOR_START_POWER);
    SoftPWMSet(MOTOR_0_PIN[1], positive_fl*MOTOR_START_POWER);

    SoftPWMSet(MOTOR_3_PIN[0], (not positive_bl)*MOTOR_START_POWER);
    SoftPWMSet(MOTOR_3_PIN[1], positive_bl*MOTOR_START_POWER);

    SoftPWMSet(MOTOR_1_PIN[0], (not positive_fr)*MOTOR_START_POWER);
    SoftPWMSet(MOTOR_1_PIN[1], positive_fr*MOTOR_START_POWER);

    SoftPWMSet(MOTOR_2_PIN[0], positive_br*MOTOR_START_POWER);
    SoftPWMSet(MOTOR_2_PIN[1], (not positive_br)*MOTOR_START_POWER);

    delayMicroseconds(10);
}

void set_motors(WheelSpeeds wheel_speeds) {
    const uint8_t front_left_power = _map(abs(wheel_speeds.front_left), 0, OMEGA_MAX, MOTOR_POWER_MIN, MOTOR_POWER_MAX);
    const uint8_t front_right_power = _map(abs(wheel_speeds.front_right), 0, OMEGA_MAX, MOTOR_POWER_MIN, MOTOR_POWER_MAX);
    const uint8_t back_right_power = _map(abs(wheel_speeds.back_right), 0, OMEGA_MAX, MOTOR_POWER_MIN, MOTOR_POWER_MAX);
    const uint8_t back_left_power = _map(abs(wheel_speeds.back_left), 0, OMEGA_MAX, MOTOR_POWER_MIN, MOTOR_POWER_MAX);

    bool positive_fl = wheel_speeds.front_left > 0;
    bool positive_bl = wheel_speeds.back_left > 0;
    bool positive_fr = wheel_speeds.back_left > 0;
    bool positive_br = wheel_speeds.back_left > 0;
    overcome_static_friction(positive_fl, positive_bl, positive_fr, positive_br);

    // Left wheels
    SoftPWMSet(MOTOR_0_PIN[0], (not positive_fl)*front_left_power);
    SoftPWMSet(MOTOR_0_PIN[1], positive_fl*front_left_power);

    SoftPWMSet(MOTOR_3_PIN[0], (not positive_bl)*back_left_power);
    SoftPWMSet(MOTOR_3_PIN[1], positive_bl*back_left_power);

    // Right wheels
    SoftPWMSet(MOTOR_1_PIN[0], (not positive_fr)*front_right_power);
    SoftPWMSet(MOTOR_1_PIN[1], positive_fr*front_right_power);

    SoftPWMSet(MOTOR_2_PIN[0], positive_br*back_right_power);
    SoftPWMSet(MOTOR_2_PIN[1], (not positive_br)*back_right_power);
}

void move(uint8_t angle, uint8_t power, uint8_t rot_vel) {
    WheelSpeeds wheel_speeds = mecanum_inverse_kinematics(angle, power, rot_vel);
    set_motors(wheel_speeds);
}


