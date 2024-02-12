#include "typedefs.h"
#include <Arduino.h>
#include <SoftPWM.h>

#include "car_control.h"

// https://research.ijcaonline.org/volume113/number3/pxc3901586.pdf
constexpr float R_INV = 1.0/0.03;   // reciprocal of wheel radius
constexpr float L_X = 0.11/2;       // half length of front-back distance (between wheel centers)
constexpr float L_Y = 0.16/2;       // half length of left-right distance (between wheel centers)

constexpr uint8_t MOTOR_POWER_MIN = 0;  
constexpr uint8_t MOTOR_POWER_MAX = 255; 
constexpr uint8_t MOTOR_START_POWER = 100; 
constexpr float OMEGA_MIN = -21.0; 
constexpr float OMEGA_MAX = 21.0;

//  Motor pin layout
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


// https://research.ijcaonline.org/volume113/number3/pxc3901586.pdf
WheelSpeeds mecanum_inverse_kinematics(const Action& action) {
    const float v = action.velocity;
    const float cos_phi = cos(action.angle*DEG_TO_RAD);  
    const float sin_phi = sin(action.angle*DEG_TO_RAD);
    const float omega_z = action.rot_vel;

    const float v_x = action.velocity * cos(action.angle*DEG_TO_RAD);
    const float v_y = action.velocity * sin(action.angle*DEG_TO_RAD);
    const float sum = v_x + v_y;
    const float diff = v_x - v_y;
    
    return (WheelSpeeds){
        front_left:     constrain(R_INV*(diff - (L_X + L_Y)*omega_z), OMEGA_MIN, OMEGA_MAX),
        front_right:    constrain(R_INV*(sum  + (L_X + L_Y)*omega_z), OMEGA_MIN, OMEGA_MAX),
        back_left:      constrain(R_INV*(sum  - (L_X + L_Y)*omega_z), OMEGA_MIN, OMEGA_MAX),
        back_right:     constrain(R_INV*(diff + (L_X + L_Y)*omega_z), OMEGA_MIN, OMEGA_MAX),
    };
}


// Helps for very low velocities
const void overcome_static_friction(const bool positive_fl,const bool positive_bl, const bool positive_fr, const bool positive_br) {
    SoftPWMSet(MOTOR_0_PIN[0], (not positive_fl)*MOTOR_START_POWER);
    SoftPWMSet(MOTOR_0_PIN[1], positive_fl*MOTOR_START_POWER);

    SoftPWMSet(MOTOR_3_PIN[0], (not positive_bl)*MOTOR_START_POWER);
    SoftPWMSet(MOTOR_3_PIN[1], positive_bl*MOTOR_START_POWER);

    SoftPWMSet(MOTOR_1_PIN[0], positive_fr*MOTOR_START_POWER);
    SoftPWMSet(MOTOR_1_PIN[1], (not positive_fr)*MOTOR_START_POWER);

    SoftPWMSet(MOTOR_2_PIN[0], positive_br*MOTOR_START_POWER);
    SoftPWMSet(MOTOR_2_PIN[1], (not positive_br)*MOTOR_START_POWER);

    delayMicroseconds(10);
}

void set_motors(const WheelSpeeds& wheel_speeds) {
    const uint8_t front_left_power = map(abs(wheel_speeds.front_left), 0, OMEGA_MAX, MOTOR_POWER_MIN, MOTOR_POWER_MAX);
    const uint8_t back_left_power = map(abs(wheel_speeds.back_left), 0, OMEGA_MAX, MOTOR_POWER_MIN, MOTOR_POWER_MAX);
    const uint8_t front_right_power = map(abs(wheel_speeds.front_right), 0, OMEGA_MAX, MOTOR_POWER_MIN, MOTOR_POWER_MAX);
    const uint8_t back_right_power = map(abs(wheel_speeds.back_right), 0, OMEGA_MAX, MOTOR_POWER_MIN, MOTOR_POWER_MAX);

    bool positive_fl = wheel_speeds.front_left > 0;
    bool positive_bl = wheel_speeds.back_left > 0;
    bool positive_fr = wheel_speeds.front_right > 0;
    bool positive_br = wheel_speeds.back_right > 0;
    overcome_static_friction(positive_fl, positive_bl, positive_fr, positive_br);

    // Left wheels
    SoftPWMSet(MOTOR_0_PIN[0], (not positive_fl)*front_left_power);
    SoftPWMSet(MOTOR_0_PIN[1], positive_fl*front_left_power);

    SoftPWMSet(MOTOR_3_PIN[0], (not positive_bl)*back_left_power);
    SoftPWMSet(MOTOR_3_PIN[1], positive_bl*back_left_power);

    // Right wheels
    SoftPWMSet(MOTOR_1_PIN[0], positive_fr*front_right_power);
    SoftPWMSet(MOTOR_1_PIN[1], (not positive_fr)*front_right_power);

    SoftPWMSet(MOTOR_2_PIN[0], positive_br*back_right_power);
    SoftPWMSet(MOTOR_2_PIN[1], (not positive_br)*back_right_power);
}

void move(const Action& action) {
    set_motors(mecanum_inverse_kinematics(action));
}


