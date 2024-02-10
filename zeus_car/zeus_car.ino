#include <Arduino.h>
#include <SoftPWM.h>
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

#define STANDBY 0
#define ACT 1


Esp32Listener esp_listener;


void setup() {
    SoftPWMBegin();
    rgb_begin();
    rgb_write(ORANGE);
    // esp_listener.init(SSID, PASSWORD, WIFI_MODE, PORT);
    rgb_write(CYAN);
    start_motors();
    start_signal();
}


void loop() {
    // Esp32Listener::Message message = esp_listener.listen();
    // uint8_t mode = message.mode;
    // Esp32Listener::Action action = message.action;

    Esp32Listener::Action action = {
        angle: 128,
        magnitude: 3,
    };

    // uint8_t mode = 0;

    // switch (mode) {

    //     case STANDBY:
    //         standby();
    //         break;

    //     case ACT:
    //         act(action);
    //         break;
    // }

    delay(2000);
    rgb_write(GREEN); delay(20);
    WheelSpeeds wheelspeeds = {
        front_left: 0.0,
        front_right: 5.0,
        back_right: -5.0,
        back_left: 0.0,
    };  
    set_motors(wheelspeeds);
    delay(2000);
    rgb_write(ORANGE); delay(20);
    wheelspeeds = {
        front_left: -5.0,
        front_right: 0.0,
        back_right: 0.0,
        back_left: 5.0,
    };  
    set_motors(wheelspeeds);
}
 

void standby() {
    stop_motors();
    rgb_write(PURPLE);
    delay(1000);   
}


void act(Esp32Listener::Action action) {
    rgb_write(GREEN);
    delay(2000);   
    move(action.angle, action.magnitude, 0);
}

