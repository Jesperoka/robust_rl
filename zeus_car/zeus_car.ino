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
    Serial.begin(115200);
    // esp_listener.init(SSID, PASSWORD, WIFI_MODE, PORT);
    rgb_write(CYAN);
    start_motors();
    start_signal();
    Serial.println("Setup done");
}


void loop() {
    // Esp32Listener::Message message = esp_listener.listen();
    // uint8_t mode = message.mode;
    // Esp32Listener::Action action = message.action;

    standby();
    delay(4000);

    Esp32Listener::Action action = {
        angle: 128,
        magnitude: 15,
    };

    uint8_t mode = 1;

    switch (mode) {

        case STANDBY:
            standby();
            break;

        case ACT:
            rgb_write(BLUE);
            act(action);
            delay(1000);
            action = {
                angle: 0,
                magnitude: 15,

            };
            rgb_write(RED);
            act(action);
            delay(1000);
            action = {
                angle: 255,
                magnitude: 15,
            };
            rgb_write(GREEN);
            act(action);
            delay(1000);
            action = {
                angle: 25,
                magnitude: 15,
            };
            rgb_write(PURPLE);
            act(action);
            delay(1000);
            action = {
                angle: 50,
                magnitude: 15,
            };
            rgb_write(YELLOW);
            act(action);
            delay(1000);
            action = {
                angle: 200,
                magnitude: 15,
            };
            rgb_write(CYAN);
            act(action);
            delay(1000);
            break;
    }
}
 

void standby() {
    stop_motors();
    rgb_write(PURPLE);
    delay(1000);   
}


void act(Esp32Listener::Action action) {
    // rgb_write(GREEN);
    move(action.angle, action.magnitude, 0);
}

