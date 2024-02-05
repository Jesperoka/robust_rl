#include <Arduino.h>
#include <SoftPWM.h>
#include <esp32_listener.h>

/*
 *  [0]--|||--[1]
 *   |         |
 *   |         |
 *   |         |
 *   |         |
 *  [3]-------[2]
 */


#define MOTOR_POWER_MIN 28  // NOTE: set to 73 for 20% min power 
#define MOTOR_POWER_MAX 255 // NOTE: set to 210 for 80% max power

#define WIFI_MODE WIFI_MODE_AP
#define SSID "Zeus_Car"
#define PASSWORD "12345678"
#define PORT "8765"

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


void start_car() {
    for (uint8_t i = 0; i < 8; i++) {
        SoftPWMSet(MOTOR_PINS[i], 0);
        SoftPWMSetFadeTime(MOTOR_PINS[i], 100, 100);
    }
}

void set_motors(int8_t power0, int8_t power1, int8_t power2, int8_t power3) {
    bool dir[4];
    int8_t power[4] = { power0, power1, power2, power3 };
    int8_t newPower[4];

    for (uint8_t i = 0; i < 4; i++) {
        dir[i] = power[i] > 0;

        // TODO: why invert it?
        if (MOTOR_DIRECTIONS[i]) { 
            dir[i] = !dir[i];
        }

        if (power[i] == 0) {
            newPower[i] = 0;
        } else {
            newPower[i] = map(abs(power[i]), 0, 100, MOTOR_POWER_MIN, MOTOR_POWER_MAX);
        }
        SoftPWMSet(MOTOR_PINS[i * 2], dir[i] * newPower[i]);
        SoftPWMSet(MOTOR_PINS[i * 2 + 1], !dir[i] * newPower[i]);
    }
}

// TODO: make standby functionality
void standby() {
    stop();
    rgbWrite(PURPLE);
    delay(100);
} 

// TODO: make action functionality
void act() {
    move(angle, power, remoteHeading);
    rgbWrite(GREEN);
} 

