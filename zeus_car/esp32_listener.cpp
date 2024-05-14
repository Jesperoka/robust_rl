#include "string.h"
#include "stdlib.h"
#include "stdint.h"
#include "esp32_listener.h"
#include "typedefs.h"
#include "rgb.h"

constexpr unsigned char SERIAL_TIMEOUT = 100; // Assuming the timeout will not exceed 255
constexpr uint8_t SERIAL_WRITE_MAX_RETRIES = 3;


// Constants for communication with ESP32-CAM firmware
// --------------------------------------------------
#define OK_FLAG "[OK]"
#define WS_HEADER "WS+"
#define OK_FLAG_END_MARKER ']'
#define HEADER_START_MARKER 'W'
#define HEADER_END_MARKER '+'
#define COMMAND_END_MARKER ';'
#define DELIMITER ";"
#define MIN_VALID_MESSAGE_LENGTH 11
// --------------------------------------------------
// #define WIFI_MODE_NONE "0"
// #define WIFI_MODE_STA "1"
// #define WIFI_MODE_AP "2"
// --------------------------------------------------

#define starts_with(str, prefix) (strncmp(str, prefix, strlen(prefix)) == 0)
#define contains(string, substring) (strstr(string, substring) != NULL)


void Esp32Listener::init(const char* ssid, const char* password, const char* wifi_mode, const char* ws_port) {
    Serial.begin(115200);

    uint32_t cmd_timeout = 3000; 
    Buffer version = write_serial("RESET", "", cmd_timeout);
    Serial.print(F("ESP32 firmware version: ")); 
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
    Buffer rx_buffer = read_serial(HEADER_START_MARKER, COMMAND_END_MARKER);

    Message default_message = Message{
        action: (Action){
            velocity:   0.0,
            angle:      0.0,
            rot_vel:    0.0,
        }, 
        mode: Mode::CONTINUE,
    }; 

    if (rx_buffer.length < MIN_VALID_MESSAGE_LENGTH) {return default_message;}
    if (!starts_with(rx_buffer.data, WS_HEADER)) {return default_message;}

    rx_buffer = strip_to_and_including_last(rx_buffer, HEADER_END_MARKER);

    return parse_message(rx_buffer, DELIMITER);
}


Message Esp32Listener::parse_message(Buffer& rx_buffer, const char* delimiter) {
    Message message;

    char* token = strtok(rx_buffer.data, delimiter);
    message.action.velocity = (token != NULL) ? atof(token) : 0.0;

    token = strtok(NULL, delimiter);
    message.action.angle = (token != NULL) ? atof(token) : 0.0;

    token = strtok(NULL, delimiter);
    message.action.rot_vel = (token != NULL) ? atof(token) : 0.0;

    token = strtok(NULL, delimiter);
    message.mode = (token != NULL) ? (Mode)atoi(token) : Mode::CONTINUE;

    return message;
}

// TODO: document
// Reads from start_marker until two consequtive occurences of end_marker.
// So "WS+0.8;1.32;-0.1;1;;;;;;;;;;;;;" becomes "WS+0.8;1.32;-0.1;1;" if start_marker is 'W' and end_marker is ';'.
Esp32Listener::Buffer Esp32Listener::read_serial(const char start_marker, const char end_marker) {
    Buffer rx_buffer = {data: {}, length: 0};
    bool in_progress = false;
    
    char prev_character = end_marker + 1; // just to make sure we don't start at the same character.
    while (Serial.available() > 0 and rx_buffer.length < SERIAL_BUFFER_SIZE - 1) {
        char character = Serial.read();

        if (character == start_marker) { 
            in_progress = true; 
        }

        if (in_progress and !(character == end_marker and prev_character == end_marker)) {
            rx_buffer.data[rx_buffer.length] = character;
            rx_buffer.length++;
        }

        if (in_progress and character == end_marker and prev_character == end_marker) {
            rx_buffer.data[rx_buffer.length] = '\0';
            return rx_buffer;
        }
        prev_character = character;
    }
    rx_buffer.data[rx_buffer.length] = '\0';

    return rx_buffer;
}


Esp32Listener::Buffer Esp32Listener::read_serial_by_size(const uint8_t size) {
    Buffer rx_buffer = {data: {}, length: 0};
    
    while (Serial.available() > 0 and rx_buffer.length < size - 1) {
        rgb_write(ORANGE);
        char character = Serial.read();
        rx_buffer.data[rx_buffer.length] = character;
        rx_buffer.length++;
    }
    rx_buffer.data[rx_buffer.length] = '\0';

    return rx_buffer;
}

/**
 * @param buffer_1 [0, 1, 2, 3, 4] (length = 5)
 * @param buffer_2 [5, 6, 7, 8, 9] (length = 5)
 * @return output_buffer [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (length = 10)
*/
Esp32Listener::Buffer Esp32Listener::join_buffers(Buffer& buffer_1, Buffer& buffer_2) {
    Buffer output_buffer = {data: {}, length: 0};
    for (uint8_t i = 0; i < buffer_1.length; i++) {
        output_buffer.data[i] = buffer_1.data[i];
        output_buffer.length++;
    }
    for (uint8_t i = 0; i < buffer_2.length; i++) {
        output_buffer.data[buffer_1.length + i] = buffer_2.data[i];
        output_buffer.length++;
    }
    output_buffer.data[output_buffer.length] = '\0';

    return output_buffer;
}


/**
 * @brief Send command to ESP32-CAM with serial
 * @param command command keyword: "TYPE", "NAME", "SSID", "PSK", "MODE", "PORT", "START", "RESET", "WS+", "AI+" 
 * @param value 
 * @return buffer of returned information on the serial port
 */
Esp32Listener::Buffer Esp32Listener::write_serial(const char* command, const char* value, uint32_t cmd_timeout) {
    for (uint8_t retry_count = 0; retry_count < SERIAL_WRITE_MAX_RETRIES; retry_count++) {
        if (retry_count == 0) {
            Serial.print(F("SET+"));
            Serial.print(command);
            Serial.println(value);
            Serial.println(F("..."));
        }

        uint32_t start_time = millis();
        while ((millis() - start_time) < cmd_timeout) {
            Buffer rx_buffer = read_serial_by_size(SERIAL_BUFFER_SIZE);

            if (contains(rx_buffer.data, OK_FLAG)) {
                rx_buffer = strip_to_and_including_last(rx_buffer, OK_FLAG_END_MARKER);
                Serial.println(F(OK_FLAG));

                return rx_buffer;
            }
        }
    }
    Serial.println(F("[FAIL]"));

    return Buffer{};
}

Esp32Listener::Buffer& Esp32Listener::strip_to_and_including_first(Buffer& rx_buffer, const char* strip_character) {
    char *strip_character_ptr = strchr(rx_buffer.data, strip_character);

    if (strip_character_ptr != NULL) {
        char *body_start = strip_character_ptr + 1;
        size_t body_length = strlen(body_start);
        memmove(rx_buffer.data, body_start, body_length + 1);
        rx_buffer.length = body_length;
    }

    return rx_buffer;
}

Esp32Listener::Buffer& Esp32Listener::strip_to_and_including_last(Buffer& rx_buffer, const char* strip_character) {
    char *strip_character_ptr = strrchr(rx_buffer.data, strip_character); // here's the difference

    if (strip_character_ptr != NULL) {
        char *body_start = strip_character_ptr + 1;
        size_t body_length = strlen(body_start);
        memmove(rx_buffer.data, body_start, body_length + 1);
        rx_buffer.length = body_length;
    }

    return rx_buffer;
}


