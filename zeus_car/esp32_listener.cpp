#include "esp32_listener.h"

#define SERIAL_TIMEOUT 100
#define WS_BUFFER_SIZE 200

// Some keywords for communication with ESP32-CAM
#define CHECK "SC"
#define OK_FLAG "[OK]"
#define ERROR_FLAG "[ERR]"
#define WS_HEADER "WS+"
#define CAM_INIT "[Init]"
#define WS_CONNECT "[CONNECTED]"
#define WS_DISCONNECT "[DISCONNECTED]"
#define APP_STOP "[APPSTOP]"

#define WIFI_MODE_NONE "0"
#define WIFI_MODE_STA "1"
#define WIFI_MODE_AP "2"

// TODO: remove these and use string.h functions
// Functions for manipulating strings
#define IsStartWith(str, prefix) (strncmp(str, prefix, strlen(prefix)) == 0)
#define StrAppend(str, suffix)                                                 \
    uint32_t len = strlen(str);                                                  \
    str[len] = suffix;                                                           \
    str[len + 1] = '\0'
#define StrClear(str) str[0] = 0

// Communication globals // NOTE: why global // TODO: remove these and use class members or pass as parameters
int32_t ws_send_time = millis();
int32_t ws_send_interval = 60;


Esp32Listener::Esp32Listener(const char* ssid, const char* password, const char* wifi_mode, const char* ws_port) {
#ifdef AI_CAM_DEBUG_CUSTOM
    Serial.begin(115200);
#endif
    char ip[25];
    char version[25];
    uint32_t cmd_timeout = 3000;

    get("RESET", version, cmd_timeout);
    Serial.print(F("ESP32 firmware version "));
    Serial.println(version);

    cmd_timeout = 1000;
    set("TYPE", "custom", cmd_timeout);
    set("NAME", "my_zeus_car", cmd_timeout);
    set("SSID", ssid, cmd_timeout);
    set("PSK", password, cmd_timeout);
    set("MODE", wifi_mode, cmd_timeout);
    set("PORT", ws_port, cmd_timeout);

    cmd_timeout = 5000;
    get("START", ip);
    delay(20);
    Serial.print(F("WebServer started on ws://"));

    Serial.print(F(":"));
    Serial.println(ws_port);
}


// Receive and process serial port data in a loop
void Esp32Listener::loop() {
    // TODO: rewrite read_into/read_serial
    read_into(recvBuffer);

    if (strlen(recvBuffer) != 0) {
        // Serial.print("recv: ");Serial.println(recvBuffer);

        // ESP32-CAM reboot detection
        if (IsStartWith(recvBuffer, CAM_INIT)) {
            Serial.println(F("ESP32-CAM reboot detected"));
            carStop();
            ws_connected = false; // first use
        }
        // ESP32-CAM websocket connected
        else if (IsStartWith(recvBuffer, WS_CONNECT)) {
            Serial.println(F("ESP32-CAM websocket connected"));
            ws_connected = true;
        }
        // ESP32-CAM websocket disconnected
        else if (IsStartWith(recvBuffer, WS_DISCONNECT)) {
            Serial.println(F("ESP32-CAM websocket disconnected"));
            ws_connected = false;
        }
        // ESP32-CAM APP_STOP
        else if (IsStartWith(recvBuffer, APP_STOP)) {
            if (ws_connected) {
                Serial.println(F("APP STOP"));
            }
            ws_connected = false;
        }

        // NOTE: this is where the actual received data gets handeled

        // recv WS+ data
        else if (IsStartWith(recvBuffer, WS_HEADER)) {
            Serial.print("RX:");
            Serial.println(recvBuffer);
            ws_connected = true;
            this->substring(recvBuffer, strlen(WS_HEADER));

        }
        // send data
        if (millis() - ws_send_time > ws_send_interval) {
            this->send_data();
            ws_send_time = millis();
        }
    }
}

void Esp32Listener::read_serial(char* buffer) {
    // TODO: rewrite
}

// websocket received data processing
void on_receive() {
    // TODO: create Message and Action structs and a Mode enum
    Message message = esp_listener.received(); // TODO: figure out what to name
    Mode mode = message.mode;
    Action action = message.action;

    if (mode == ACT) {
        act();
        return; // NOTE: want to return mode?
    }

    if (mode == STANDBY) {
        standby();
        return; // NOTE: want to return mode?
    }
}


// Send data on serial port, prepends header (WS_HEADER)
void Esp32Listener::send_data() {
    Serial.print(F(WS_HEADER));
    serializeJson(send_doc, Serial);
    Serial.print("\n");
    Serial.flush();
}

void Esp32Listener::set_command_timeout(uint32_t timeout) {
    cmd_timeout = timeout;
}


/**
 * @brief Send command to ESP32-CAM with serial
 * @param command command keyword
 * @param value
 * @param result returned information from serial
 */
void Esp32Listener::command(const char *command, const char *value, char *result, uint32_t cmd_timeout = SERIAL_TIMEOUT) {
    bool is_ok = false;
    uint8_t retry_count = 0;
    uint8_t retry_max_count = 3;

    while (!is_ok && retry_count < retry_max_count) {
        if (retry_count == 0) {
            Serial.print(F("SET+"));
            Serial.print(command);
            Serial.println(value);
            Serial.print(F("..."));
        }
        retry_count++;

        uint32_t start_time = millis();
        while ((millis() - start_time) < cmd_timeout) {
            read_into(recvBuffer);

            if (IsStartWith(recvBuffer, OK_FLAG)) {
                is_ok = true;
                Serial.println(F(OK_FLAG));
                this->substring(recvBuffer, strlen(OK_FLAG) + 1); // Add 1 for Space
                                                                  // !!! Note that the reslut size here is too small and may be out of
                                                                  // bounds, causing unexpected data changes
                strcpy(result, recvBuffer);
                break;
            }
        }
    }

    if (is_ok == false) {
        Serial.println(F("[FAIL]"));
        delay(10000)
    }
}

// TODO: rename command and remove these wrappers

// Use the command() function to set up the ESP32-CAM
void Esp32Listener::set(const char *command, uint32_t cmd_timeout) {
    char result[10];
    command(command, "", result, cmd_timeout);
}

// Use the command() function to set up the ESP32-CAM
void Esp32Listener::set(const char *command, const char *value, uint32_t cmd_timeout) {
    char result[10];
    command(command, value, result, cmd_timeout);
}

// Use the comand() function to set up the ESP32-CAM and receive return information
void Esp32Listener::get(const char *command, char *result) {
    command(command, "", result);
}

// Use the comand() function to set up the ESP32-CAM and receive return information
void Esp32Listener::get(const char *command, const char *value, char *result, uint32_t cmd_timeout) {
    command(command, value, result, cmd_timeout);
}



void Esp32Listener::setValue(uint8_t region, double value) {
    setStrOf(recvBuffer, region, String(value));
}

/**
 * @brief subtract part of the string
 * @param buf string pointer to be subtract
 * @param start start position of content to be subtracted
 * @param end end position of Content to be subtracted
 */
// TODO: replace with String().substring() or use C string.h functions
void Esp32Listener::substring(char *str, int16_t start, int16_t end) {
    uint8_t length = strlen(str);
    if (end == -1) {
        end = length;
    }
    for (uint8_t i = 0; i < end; i++) {
        if (i + start < end) {
            str[i] = str[i + start];
        } else {
            str[i] = '\0';
        }
    }
}

/**
 * @brief Split the string by a cdivider,
 *         and return characters of the selected index
 *
 * @param buf string pointer to be split
 * @param index which index do you wish to return
 * @param result char array pointer to hold the result
 * @param divider
 */
// TODO: definetly consider using native String functions or C string.h
void Esp32Listener::getStrOf(char *str, uint8_t index, char *result, char divider) {
    uint8_t start, end;
    uint8_t length = strlen(str);
    uint8_t i, j;

    // Get start index
    if (index == 0) {
        start = 0;
    } else {
        for (start = 0, j = 1; start < length; start++) {
            if (str[start] == divider) {
                if (index == j) {
                    start++;
                    break;
                }
                j++;
            }
        }
    }
    // Get end index
    for (end = start, j = 0; end < length; end++) {
        if (str[end] == divider) {
            break;
        }
    }
    // Copy result
    // if ((end - start + 2) > sizeof(result)) { // '\0' takes up one byte
    //   end = start + sizeof(result) -1;
    // }

    for (i = start, j = 0; i < end; i++, j++) {
        result[j] = str[i];
    }
    result[j] = '\0';
}

/**
 * @brief split by divider, filling the value to a position in the string
 * @param str string pointer to be operated
 * @param index which index do you wish to return
 * @param value the value to be filled
 * @param divider
 */
// TODO: investigate strtok() and other C string.h functions for this
void Esp32Listener::setStrOf(char *str, uint8_t index, String value,
        char divider = ';') {
    uint8_t start, end;
    uint8_t length = strlen(str);
    uint8_t i, j;
    // Get start index
    if (index == 0) {
        start = 0;
    } else {
        for (start = 0, j = 1; start < length; start++) {
            if (str[start] == divider) {
                if (index == j) {
                    start++;
                    break;
                }
                j++;
            }
        }
    }
    // Get end index
    for (end = start, j = 0; end < length; end++) {
        if (str[end] == divider) {
            break;
        }
    }
    String strString = str;
    String strValue =
        strString.substring(0, start) + value + strString.substring(end);
    strcpy(str, strValue.c_str());
}
