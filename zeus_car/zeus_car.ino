#include <Arduino.h>
#include <SoftPWM.h>
#include <esp32_listener.h>
#include <car_control.h>

//  Motor layout
//
//  [0]--|||--[1]
//   |         |
//   |         |
//   |         |
//   |         |
//  [3]-------[2]

// Constants

#define WIFI_MODE WIFI_MODE_AP
#define SSID "Zeus_Car"
#define PASSWORD "12345678"
#define PORT "8765"

#define STANDBY 0
#define ACT 0

// Globals
Esp32Listener esp_listener = Esp32Listener(SSID, PASSWORD, WIFI_MODE, PORT);


void setup() {
    Serial.begin(115200);
    Serial.println("Initializing zeus_car");
    SoftPWMBegin();
    rgb_begin();
    rgb_write(ORANGE);
    start_car();
    esp_listener.begin(SSID, PASSWORD, WIFI_MODE, PORT);
    esp_listener.set_on_receive(on_receive);
}

// Program loop
void loop() {
    Message message = esp_listener.listen();
    Mode mode = message.mode;
    Action action = message.action;

    switch (mode) {

        case STANDBY:
            standby();
            break;

        case ACT:
            act(action);
            break;
    }
}


// TODO: make standby functionality
void standby() {
    stop();
    rgbWrite(PURPLE);
    delay(100);
}

// TODO: make action functionality
void act(Action action) {
    move(action.angle, action.magnitude);
    rgbWrite(GREEN);
}

