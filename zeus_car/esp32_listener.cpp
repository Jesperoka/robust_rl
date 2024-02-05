#include "esp32_listener.h"

#define DateSerial Serial
#define DebugSerial Serial

#define SERIAL_TIMEOUT 100
#define WS_BUFFER_SIZE 200
#define CHAR_TIMEOUT 50

// Some keywords for communication with ESP32-CAM
#define CHECK "SC"
#define OK_FLAG "[OK]"
#define ERROR_FLAG "[ERR]"
#define WS_HEADER "WS+"
#define CAM_INIT "[Init]"
#define WS_CONNECT "[CONNECTED]"
#define WS_DISCONNECT "[DISCONNECTED]"
#define APP_STOP "[APPSTOP]"

/**
 * @name Set the print level of information received by esp32-cam
 *
 * @code {.cpp}
 * #define CAM_DEBUG_LEVEL CAM_DEBUG_LEVEL_INFO
 * @endcode
 *
 */
#define CAM_DEBUG_LEVEL CAM_DEBUG_LEVEL_INFO
#define CAM_DEBUG_LEVEL_OFF 0
#define CAM_DEBUG_LEVEL_ERROR 1
#define CAM_DEBUG_LEVEL_INFO 2
#define CAM_DEBUG_LEVEL_DEBUG 3
#define CAM_DEBUG_LEVEL_ALL 4

#define CAM_DEBUG_HEAD_ALL "[CAM_A]"
#define CAM_DEBUG_HEAD_ERROR "[CAM_E]"
#define CAM_DEBUG_HEAD_INFO "[CAM_I]"
#define CAM_DEBUG_HEAD_DEBUG "[CAM_D]"

/**
 * @name Define component-related values
 */
#define DPAD_STOP 0
#define DPAD_FORWARD 1
#define DPAD_BACKWARD 2
#define DPAD_LEFT 3
#define DPAD_RIGHT 4

#define JOYSTICK_X 0
#define JOYSTICK_Y 1
#define JOYSTICK_ANGLE 2
#define JOYSTICK_RADIUS 3

#define WIFI_MODE_NONE "0"
#define WIFI_MODE_STA "1"
#define WIFI_MODE_AP "2"

#define REGION_A 0
#define REGION_B 1
#define REGION_C 2
#define REGION_D 3
#define REGION_E 4
#define REGION_F 5
#define REGION_G 6
#define REGION_H 7
#define REGION_I 8
#define REGION_J 9
#define REGION_K 10
#define REGION_L 11
#define REGION_M 12
#define REGION_N 13
#define REGION_O 14
#define REGION_P 15
#define REGION_Q 16
#define REGION_R 17
#define REGION_S 18
#define REGION_T 19
#define REGION_U 20
#define REGION_V 21
#define REGION_W 22
#define REGION_X 23
#define REGION_Y 24
#define REGION_Z 25


// Functions for manipulating strings
#define IsStartWith(str, prefix) (strncmp(str, prefix, strlen(prefix)) == 0)
#define StrAppend(str, suffix)                                                 \
    uint32_t len = strlen(str);                                                  \
    str[len] = suffix;                                                           \
    str[len + 1] = '\0'
#define StrClear(str) str[0] = 0

// Communication globals // NOTE: why global
int32_t cmd_timeout = SERIAL_TIMEOUT;
int32_t ws_send_time = millis();
int32_t ws_send_interval = 60;


Esp32Listener::Esp32Listener(const char* ssid, const char* password, const char* wifi_mode, const char* ws_port) {
#ifdef AI_CAM_DEBUG_CUSTOM
    DateSerial.begin(115200);
#endif
    char ip[25];
    char version[25];

    set_command_timeout(3000);
    this->get("RESET", version);
    DebugSerial.print(F("ESP32 firmware version "));
    DebugSerial.println(version);

    set_command_timeout(1000);
    this->set("TYPE", "custom");
    this->set("NAME", "my_zeus_car");
    this->set("SSID", ssid);
    this->set("PSK", password);
    this->set("MODE", wifi_mode);
    this->set("PORT", ws_port);

    set_command_timeout(5000);
    this->get("START", ip);
    delay(20);
    DebugSerial.print(F("WebServer started on ws://"));

    DebugSerial.print(F(":"));
    DebugSerial.println(ws_port);

    set_command_timeout(SERIAL_TIMEOUT);
}


// Receive and process serial port data in a loop
void Esp32Listener::loop() {
    this->read_into(recvBuffer);
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
            DateSerial.print("RX:");
            DateSerial.println(recvBuffer);
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

// websocket received data processing
void on_receive() {
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


/**
 * @brief Print the information received from esp32-CAm,
 *        according to the set of CAM_DEBUG_LEVEL
 *
 * @param msg Message to be detected
 */
void EspEsp32Listener::debug(char *msg) {
#if (CAM_DEBUG_LEVEL == CAM_DEBUG_LEVEL_ALL) // all
    DebugSerial.print(CAM_DEBUG_HEAD_ALL);
    DebugSerial.println(msg);
#elif (CAM_DEBUG_LEVEL == CAM_DEBUG_LEVEL_ERROR) // error
    if (IsStartWith(msg, CAM_DEBUG_HEAD_ERROR)) {
        DebugSerial.println(msg);
    }
#elif (CAM_DEBUG_LEVEL == CAM_DEBUG_LEVEL_INFO)  // info
    if (IsStartWith(msg, CAM_DEBUG_HEAD_ERROR)) {
        DebugSerial.println(msg);
    } else if (IsStartWith(msg, CAM_DEBUG_HEAD_INFO)) {
        DebugSerial.println(msg);
    }
#elif (CAM_DEBUG_LEVEL == CAM_DEBUG_LEVEL_DEBUG) // debug
    if (IsStartWith(msg, CAM_DEBUG_HEAD_ERROR)) {
        DebugSerial.println(msg);
    } else if (IsStartWith(msg, CAM_DEBUG_HEAD_INFO)) {
        DebugSerial.println(msg);
    } else if (IsStartWith(msg, CAM_DEBUG_HEAD_DEBUG)) {
        DebugSerial.println(msg);
    }
#endif
}

/**
 * @brief Store the data read from the serial port into the buffer
 *
 * @param buffer  Pointer to the String value of the stored data
 */
void Esp32Listener::read_into(char *buffer) {
    /* !!! attention buffer size*/
    bool finished = false;
    char inchar;
    StrClear(buffer);
    uint32_t count = 0;

    uint32_t char_time = millis();

    // recv Byte
    while (DateSerial.available()) {
        count += 1;
        if (count > WS_BUFFER_SIZE) {
            finished = true;
            break;
        }
        inchar = (char)DateSerial.read();
        // Serial.print(inchar);
        if (inchar == '\n') {
            finished = true;
            // Serial.println(">");
            break;
        } else if (inchar == '\r') {
            continue;
        } else if ((int)inchar > 31 && (int)inchar < 127) {
            StrAppend(buffer, inchar);
            delay(1); // Wait for StrAppend
        }
    }

    // if recv debug info
    if (finished) {
        debug(buffer);
        if (IsStartWith(buffer, CAM_DEBUG_HEAD_DEBUG)) {
#if (CAM_DEBUG_LEVEL == CAM_DEBUG_LEVEL_DEBUG) // all
            DebugSerial.print(CAM_DEBUG_HEAD_DEBUG);
            DebugSerial.println(buffer);
#endif
            StrClear(buffer);
        }
    }
}

// TODO: look through commit histrory to understand what the param is for
// ------------------------------------------------------------
// ------------------------------------------------------------
// ------------------------------------------------------------

/**
 * @brief Serial port sends data, automatically adds header (WS_HEADER)
 *
 * @param sendBuffer  Pointer to the character value of the data buffer to be
 * sent
 */
void Esp32Listener::send_data() {
    DateSerial.print(F(WS_HEADER));
    serializeJson(send_doc, DateSerial);
    DateSerial.print("\n");
    DateSerial.flush();
}

void Esp32Listener::set_command_timeout(uint32_t _timeout) {
    cmd_timeout = _timeout;
}

// TODO: understand what this function is supposed to do
// ------------------------------------------------------------
// ------------------------------------------------------------
// ------------------------------------------------------------

/**
 * @brief Send command to ESP32-CAM with serial
 *
 * @param command command keyword
 * @param value
 * @param result returned information from serial
 */
void Esp32Listener::command(const char *command, const char *value, char *result) {
    bool is_ok = false;
    uint8_t retry_count = 0;
    uint8_t retry_max_count = 3;

    while (retry_count < retry_max_count) {
        if (retry_count == 0) {
            DateSerial.print(F("SET+"));
            DateSerial.print(command);
            DateSerial.println(value);
            DateSerial.print(F("..."));
        }
        retry_count++;

        uint32_t start_time = millis();
        while ((millis() - start_time) < cmd_timeout) {
            this->read_into(recvBuffer);
            if (IsStartWith(recvBuffer, OK_FLAG)) {
                is_ok = true;
                DateSerial.println(F(OK_FLAG));
                this->substring(recvBuffer, strlen(OK_FLAG) + 1); // Add 1 for Space
                                                                  // !!! Note that the reslut size here is too small and may be out of
                                                                  // bounds, causing unexpected data changes
                strcpy(result, recvBuffer);
                break;
            }
        }

        if (is_ok == true) {
            break;
        }
    }

    if (is_ok == false) {
        Serial.println(F("[FAIL]"));
        while (1)
            ;
    }
}

/**
 * @brief Use the comand() function to set up the ESP32-CAM
 *
 * @param command command keyword
 */
void Esp32Listener::set(const char *command) {
    char result[10];
    this->command(command, "", result);
}

/**
 * @brief Use the comand() function to set up the ESP32-CAM
 *
 * @param command command keyword
 * @param value
 *
 * @code {.cpp}
 * set("NAME", "Zeus_Car");
 * set("TYPE", "Zeus_Car");
 * set("SSID", "Zeus_Car");
 * set("PSK",  "12345678");
 * set("MODE", WIFI_MODE_AP);
 * set("PORT", "8765");
 * @endcode
 *
 */
void Esp32Listener::set(const char *command, const char *value) {
    char result[10];
    this->command(command, value, result);
}

/**
 * @brief Use the comand() function to set up the ESP32-CAM,
 *        and receive return information
 *
 * @param command command keyword
 * @param value
 * @param result returned information from serial
 * @code {.cpp}
 * char ip[15];
 * get("START", ip);
 * @endcode
 */
void Esp32Listener::get(const char *command, char *result) {
    this->command(command, "", result);
}

/**
 * @brief Use the comand() function to set up the ESP32-CAM,
 *        and receive return information
 *
 * @param command command keyword
 * @param value
 * @param result returned information from serial
 */
void Esp32Listener::get(const char *command, const char *value, char *result) {
    this->command(command, value, result);
}


/**
 * @brief Interpret the value of the Joystick component from the buf string
 *
 * @param buf string pointer to be interpreted
 * @param region the key of component
 * @param axis which type of value that you want,
 *             could be JOYSTICK_X, JOYSTICK_Y, JOYSTICK_ANGLE, JOYSTICK_RADIUS
 * @return the value of the Joystick component
 */
int16_t Esp32Listener::getJoystick(uint8_t region, uint8_t axis) {
    char valueStr[20];
    int16_t x, y, angle, radius;
    getStrOf(recvBuffer, region, valueStr, ';');
    x = getIntOf(valueStr, 0, ',');
    y = getIntOf(valueStr, 1, ',');
    angle = atan2(x, y) * 180.0 / PI;
    radius = sqrt(y * y + x * x);
    switch (axis) {
        case JOYSTICK_X:
            return x;
        case JOYSTICK_Y:
            return y;
        case JOYSTICK_ANGLE:
            return angle;
        case JOYSTICK_RADIUS:
            return radius;
        default:
            return 0;
    }
}

/**
 * @brief Interpret the value of the DPad component from the buf string
 *
 * @param buf string pointer to be interpreted
 * @param region the key of component
 *
 * @return the value of the DPadDPad component,
 *         it could be null, "forward", "backward", "left", "stop"
 */
uint8_t Esp32Listener::getDPad(uint8_t region) {
    char value[20];
    getStrOf(recvBuffer, region, value, ';');
    uint8_t result;
    if ((String)value == (String) "forward")
        result = DPAD_FORWARD;
    else if ((String)value == (String) "backward")
        result = DPAD_BACKWARD;
    else if ((String)value == (String) "left")
        result = DPAD_LEFT;
    else if ((String)value == (String) "right")
        result = DPAD_RIGHT;
    else if ((String)value == (String) "stop")
        result = DPAD_STOP;
    return result;
}


void Esp32Listener::setValue(uint8_t region, double value) {
    setStrOf(recvBuffer, region, String(value));
}

// TODO: Understand all functions below and move out of class.
// TODO: revamp some of the functions below if needed.
// ------------------------------------------------------------
// ------------------------------------------------------------
// ------------------------------------------------------------

/**
 * @brief subtract part of the string
 *
 * @param buf string pointer to be subtract
 * @param start start position of content to be subtracted
 * @param end end position of Content to be subtracted
 */
// TODO: replace with String().substring()
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
// TODO: definetly consider using native String functions
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
 *
 * @param str string pointer to be operated
 * @param index which index do you wish to return
 * @param value the value to be filled
 * @param divider
 */
// TODO: definetly consider using native String functions
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

/**
 * @brief Split the string by a cdivider,
 *         and return characters of the selected index.
 *         Further, the content is converted to int type.
 *
 * @param buf string pointer to be split
 * @param index which index do you wish to return
 * @param divider
 */
// TODO: definetly consider using native Int functions
int16_t Esp32Listener::getIntOf(char *str, uint8_t index, char divider = ';') {
    int16_t result;
    char strResult[20];
    getStrOf(str, index, strResult, divider);
    result = String(strResult).toInt();
    return result;
}

// TODO: definetly consider using native Bool functions
bool Esp32Listener::getBoolOf(char *str, uint8_t index) {
    char strResult[20];
    getStrOf(str, index, strResult, ';');
    return String(strResult).toInt();
}


// TODO: definetly consider using native Double functions
double Esp32Listener::getDoubleOf(char *str, uint8_t index) {
    double result;
    char strResult[20];
    getStrOf(str, index, strResult, ';');
    result = String(strResult).toDouble();
    return result;
}
