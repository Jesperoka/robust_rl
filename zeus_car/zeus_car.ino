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

Esp32Listener esp_listener = Esp32Listener(SSID, PASSWORD, WIFI_MODE, PORT);


void setup() {
    Serial.begin(115200);
    Serial.println("Initializing zeus_car");
    SoftPWMBegin();
    rgb_begin();
    rgb_write(ORANGE);
    start_motors();
}


void loop() {
    // Esp32Listener::Message message = esp_listener.listen();
    // uint8_t mode = message.mode;
    // Esp32Listener::Action action = message.action;

    Esp32Listener::Action action = {
        angle: 128,
        magnitude: 0,
    };

    uint8_t mode = 1;

    switch (mode) {

        case STANDBY:
            standby();
            break;

        case ACT:
            act(action);
            break;
    }

    delay(1000);
}
 

void standby() {
    stop_motors();
    rgb_write(PURPLE);
}


void act(Esp32Listener::Action action) {
    move(action.angle, action.magnitude, 0);
    rgb_write(GREEN);
}


