#ifndef __ESP_32_LISTENER_H__
#define __ESP_32_LISTENER_H__

#include <Arduino.h>
#include <ArduinoJson.h>

static const unsigned char WS_BUFFER_SIZE = 200; // Assuming the buffer size will not exceed 255

class Esp32Listener {
    public:
        bool ws_connected = false;
        char rx_buffer[WS_BUFFER_SIZE];
        StaticJsonDocument<WS_BUFFER_SIZE> send_doc; // NOTE: this is the same size as the receive buffer, is that a coincidence?

        Esp32Listener(const char *ssid, const char *password, const char *wifi_mode, const char *ws_port);

        const char* read_serial(const uint8_t buffer_length, const char start_marker, const char end_marker);
        void loop(); // TODO: rename
        void send_data();

        void set(const char *command);
        void set(const char *command, const char *value);
        void get(const char *command, char *result);
        void get(const char *command, const char *value, char *result);

    private:
        void command(const char *command, const char *value, char *result); // TODO: rename
        void substring(char *str, int16_t start, int16_t end = -1);

        // TODO: rename after rewrite
        void getStrOf(char *str, uint8_t index, char *result, char divider);
        void setStrOf(char *str, uint8_t index, String value, char divider = ';');
};

#endif // __ESP_32_LISTENER_H__
