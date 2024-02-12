#include "string.h"
#include "stdlib.h"
#include "stdint.h"
#include "esp32_listener.h"
#include "typedefs.h"

constexpr unsigned char SERIAL_TIMEOUT = 100; // Assuming the timeout will not exceed 255
constexpr uint8_t SERIAL_WRITE_MAX_RETRIES = 3;


// Keywords for communication with ESP32-CAM firmware
// --------------------------------------------------
#define START_MARKER '<'
#define END_MARKER '>'
#define DELIMITER ","
// --------------------------------------------------
#define OK_FLAG "[OK]"
#define WS_HEADER "WS+"

// #define WIFI_MODE_NONE "0"
// #define WIFI_MODE_STA "1"
// #define WIFI_MODE_AP "2"
// --------------------------------------------------

#define starts_with(str, prefix) (strncmp(str, prefix, strlen(prefix)) == 0)


void Esp32Listener::init(const char* ssid, const char* password, const char* wifi_mode, const char* ws_port) {
    Serial.begin(115200);

    uint32_t cmd_timeout = 3000; 
    Buffer version = write_serial("RESET", "", cmd_timeout);
    Serial.print(F("ESP32 firmware version "));
    Serial.println(version.data);

    cmd_timeout = 1000;
    write_serial("TYPE", "custom", cmd_timeout);
    write_serial("NAME", "my_zeus_car", cmd_timeout);
    write_serial("SSID", ssid, cmd_timeout);
    write_serial("PSK", password, cmd_timeout);
    write_serial("MODE", wifi_mode, cmd_timeout);
    write_serial("PORT", ws_port, cmd_timeout);

    cmd_timeout = 5000; 
    Buffer ip_addr = write_serial("START", "", cmd_timeout);
    Serial.print(F("WebServer started on ws://"));
    Serial.print(ip_addr.data);
    Serial.print(F(":"));
    Serial.println(ws_port);
}


Message Esp32Listener::listen() {
    Buffer rx_buffer = read_serial(START_MARKER, END_MARKER);

    if (rx_buffer.length != 0 and starts_with(rx_buffer.data, WS_HEADER)) {
        return parse_message(left_strip(rx_buffer, strlen(WS_HEADER)), DELIMITER);
    }
    return Message{}; // NOTE: what to do here?
}


Message Esp32Listener::parse_message(Buffer& rx_buffer, const char* delimiter) {
    Message message;

    char* token = strtok(rx_buffer.data, delimiter);
    message.action.angle = (token != NULL) ? atof(token) : 0.0;

    token = strtok(NULL, delimiter);
    message.action.velocity = (token != NULL) ? atof(token) : 0.0;

    token = strtok(NULL, delimiter);
    message.action.rot_vel = (token != NULL) ? atof(token) : 0.0;

    token = strtok(NULL, delimiter);
    message.mode = (token != NULL) ? (Mode)atoi(token) : Mode::STANDBY;

    return message;
}

// Example message: "<WS+ 45.0 0.7 1.1>"
Esp32Listener::Buffer Esp32Listener::read_serial(const char start_marker, const char end_marker) {
    Buffer rx_buffer = {data: {}, length: 0};
    bool in_progress = false;
    
    while (Serial.available() > 0 and rx_buffer.length < WS_BUFFER_SIZE - 1) {
        char character = Serial.read();

        if (character == start_marker) { 
            in_progress = true; 
        }

        if (in_progress and (character != start_marker) and (character != end_marker)) {
            rx_buffer.data[rx_buffer.length] = character;
            rx_buffer.length++;
        }

        if (character == end_marker) {
            rx_buffer.data[rx_buffer.length] = '\0';
            return rx_buffer;
        }
    }

    return rx_buffer;
}


/**
 * @brief Send command to ESP32-CAM with serial
 * @param command command keyword: "TYPE", "NAME", "SSID", "PSK", "MODE", "PORT", "START", "RESET", "WS+", "AI+" 
 * @param value 
 * @param result returned information from serial
 */
Esp32Listener::Buffer Esp32Listener::write_serial(const char *command, const char *value, uint32_t cmd_timeout) {
    bool success = false;

    for (uint8_t retry_count = 0; retry_count < SERIAL_WRITE_MAX_RETRIES; retry_count++) {
        if (retry_count == 0) {
            Serial.print(F("SET+"));
            Serial.print(command);
            Serial.println(value);
            Serial.print(F("..."));
        }

        uint32_t start_time = millis();
        while ((millis() - start_time) < cmd_timeout) {

            Buffer rx_buffer = read_serial(START_MARKER, END_MARKER);

            if (starts_with(rx_buffer.data, OK_FLAG)) {
                Serial.println(F(OK_FLAG));

                // NOTE: I might not need this since I have START_MARKER and END_MARKER
                rx_buffer = left_strip(rx_buffer, strlen(OK_FLAG)); 
                return rx_buffer;
            }
        }
    }

    if (success == false) {
        Serial.println(F("[FAIL]"));
    }

    return Buffer{};
}


// WARNING: need to double check that this works properly
// NOTE: I might not need this since I have START_MARKER and END_MARKER
Esp32Listener::Buffer& Esp32Listener::left_strip(Buffer& rx_buffer, uint8_t to_index) {
    memmove(rx_buffer.data, rx_buffer.data + to_index, rx_buffer.length - to_index);
    return rx_buffer;
}
