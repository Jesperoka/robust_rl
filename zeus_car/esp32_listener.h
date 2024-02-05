#ifndef __ESP_32_LISTENER_H__ 
#define __ESP_32_LISTENER_H__ 

#include <Arduino.h>
#include <ArduinoJson.h>


// TODO: move all #defines to cpp file (unless they're used by multiple files,
// and if so, create constants.h file)

class Esp32Listener {
    public:
        bool ws_connected = false;
        char received_buffer[WS_BUFFER_SIZE];
        StaticJsonDocument<200> send_doc;

        Esp32Listener(const char *ssid, const char *password, const char *wifi_mode, const char *ws_port);

        void set_on_receive(void (*func)());
        void set_command_timeout(uint32_t _timeout);
        void read_into(char *buffer);
        void loop(); // TODO: rename
        void send_data();

        void debug(char *msg);

        void set(const char *command);
        void set(const char *command, const char *value);
        void get(const char *command, char *result);
        void get(const char *command, const char *value, char *result);

        int16_t getSlider(uint8_t region);
        bool getButton(uint8_t region);
        bool getSwitch(uint8_t region);
        int16_t getJoystick(uint8_t region, uint8_t axis);
        uint8_t getDPad(uint8_t region);
        int16_t getThrottle(uint8_t region);
        void setMeter(uint8_t region, double value);
        void setRadar(uint8_t region, int16_t angle, double distance);
        void setGreyscale(uint8_t region, uint16_t value1, uint16_t value2,
                uint16_t value3);
        void setValue(uint8_t region, double value);
        void getSpeech(uint8_t region, char *result);

    private:
        void command(const char *command, const char *value, char *result);
        void subString(char *str, int16_t start, int16_t end = -1);

        void getStrOf(char *str, uint8_t index, char *result, char divider);
        void setStrOf(char *str, uint8_t index, String value, char divider = ';');
        int16_t getIntOf(char *str, uint8_t index, char divider = ';');
        bool getBoolOf(char *str, uint8_t index);
        double getDoubleOf(char *str, uint8_t index);
};

#endif // __ESP_32_LISTENER_H__
