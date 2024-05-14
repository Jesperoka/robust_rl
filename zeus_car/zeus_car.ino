#include <Arduino.h>
#include <SoftPWM.h>
#include "typedefs.h"
#include "car_control.h"
#include "esp32_listener.h"
#include "rgb.h"


// #define WIFI_MODE_NONE "0"
// #define WIFI_MODE_STA "1"
#define WIFI_MODE_AP "2"

#define WIFI_MODE WIFI_MODE_AP
#define SSID "Zeus_Car"
#define PASSWORD "12345678"
#define PORT "8765"

#define LOOP_DELAY 7


#ifdef __arm__
extern "C" char* sbrk(int incr);
#else  // __ARM__
extern char *__brkval;
#endif  // __arm__

int freeMemory() {
  char top;
#ifdef __arm__
  return &top - reinterpret_cast<char*>(sbrk(0));
#elif defined(CORE_TEENSY) || (ARDUINO > 103 && ARDUINO != 151)
  return &top - __brkval;
#else  // __arm__
  return __brkval ? &top - __brkval : &top - __malloc_heap_start;
#endif  // __arm__
}


Esp32Listener esp_listener;

Action prev_action = Action{
    velocity:   0.0,
    angle:      0.0,
    rot_vel:    0.0,
};
Mode prev_mode = Mode::STANDBY;


void setup() {
    SoftPWMBegin();
    start_motors();
    rgb_begin();
    rgb_write(ORANGE);
    esp_listener.init(SSID, PASSWORD, WIFI_MODE, PORT);
    start_signal();
}


void loop() {
    uint32_t time = millis();
    auto [action, mode] = esp_listener.listen();

    switch (mode) {
        case Mode::STANDBY:
            stop_motors();
            rgb_write(PURPLE);
            prev_action = action;
            prev_mode = mode;
            break;

        case Mode::ACT:
            move(action);
            rgb_write(GREEN);
            prev_action = action;
            prev_mode = mode;
            break;

        case Mode::CONTINUE:
            if (prev_mode == Mode::ACT) {
                move(prev_action);
                rgb_write(ORANGE);
            } else if (prev_mode == Mode::STANDBY) {
                stop_motors();
                rgb_write(BLUE);
            }
            break;

        default:
            stop_motors();
            rgb_write(RED);
            prev_action = Action{
                velocity:   0.0,
                angle:      0.0,
                rot_vel:    0.0,
            };
            prev_mode = Mode::STANDBY;
            delay(1000);
            break;

    }
    while (millis() - time < LOOP_DELAY) {  }
}




