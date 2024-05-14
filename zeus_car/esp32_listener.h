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
        constexpr static uint8_t SERIAL_BUFFER_SIZE =  200;

        typedef struct {
            char data[SERIAL_BUFFER_SIZE];
            uint8_t length;
        } Buffer;

        Buffer read_serial(const char start_marker, const char end_marker);
        Buffer read_serial_by_size(const uint8_t max_size);
        Buffer join_buffers(Buffer& buffer_1, Buffer& buffer_2);
        Buffer write_serial(const char *command, const char *value, uint32_t cmd_timeout); 
        Message parse_message(Buffer& rx_buffer, const char* delimiter);
        Buffer& strip_to_and_including_first(Buffer& rx_buffer, const char* strip_character);
        Buffer& strip_to_and_including_last(Buffer& rx_buffer, const char* strip_character);
};

#endif // __ESP_32_LISTENER_H__
