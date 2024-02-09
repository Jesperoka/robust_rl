#ifndef __ESP_32_LISTENER_H__
#define __ESP_32_LISTENER_H__

#include <Arduino.h>
#include <ArduinoJson.h>


class Esp32Listener {
    public:
        typedef struct {
            uint8_t angle;
            uint8_t magnitude;
        } Action;

        typedef struct {
            Action action;
            uint8_t mode;
        } Message;

        Esp32Listener(const char *ssid, const char *password, const char *wifi_mode, const char *ws_port);

        Message listen(); 

    private:
        constexpr static uint32_t WS_SEND_INTERVAL = 60;
        constexpr static uint8_t WS_BUFFER_SIZE =  200;
        static uint32_t ws_send_time;
        // StaticJsonDocument<200> send_doc;

        typedef struct {
            char data[WS_BUFFER_SIZE];
            uint8_t length;
        } Buffer;

        Buffer read_serial(const char start_marker, const char end_marker);
        Buffer write_serial(const char *command, const char *value, uint32_t cmd_timeout); 
        void send_data(JsonDocument &json);
        const Buffer left_strip(Buffer rx_buffer, uint8_t to_index);
};

#endif // __ESP_32_LISTENER_H__
