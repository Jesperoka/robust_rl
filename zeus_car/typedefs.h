#ifndef TYPEDEFS_H
#define TYPEDEFS_H

#import "Arduino.h"

enum class Mode : uint8_t {
    STANDBY = 0,
    ACT = 1,
};

typedef struct {
    float angle;
    float velocity;
    float rot_vel;
} Action;

typedef struct {
    Action action;
    Mode mode;
} Message;

typedef struct {
    float front_left;
    float front_right;
    float back_left;
    float back_right;
} WheelSpeeds;

#endif // TYPEDEFS_H
