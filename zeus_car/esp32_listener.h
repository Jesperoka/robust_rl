#ifndef __ESP_32_LISTENER_H__
#define __ESP_32_LISTENER_H__

#include <Arduino.h>
#include <ArduinoJson.hpp>
#include "typedefs.h"


class Esp32Listener {
    public:
        void init(const char *ssid, const char *password, const char *wifi_mode, const char *ws_port);
        Message listen(); 

    private:
        constexpr static uint8_t WS_BUFFER_SIZE =  200;

        typedef struct {
            char data[WS_BUFFER_SIZE];
            uint8_t length;
        } Buffer;

        Buffer read_serial(const char start_marker, const char end_marker);
        Buffer write_serial(const char *command, const char *value, uint32_t cmd_timeout); 
        Message parse_message(Buffer& rx_buffer, const char* delimiter);
        Buffer& strip_header(Buffer& rx_buffer, const char* delimiter);
};

#endif // __ESP_32_LISTENER_H__
