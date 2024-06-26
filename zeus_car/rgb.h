#ifndef __RGB_H__
#define __RGB_H__

#include <Arduino.h>

// Indicate whether RGB leds are common-anodes or not
//  0 common cathode, 1 common anode
#define COMMON_ANODE 0

#define RGB_PINS (uint8_t[3]){12, 13, 11}

#define RED           0xFF0202
#define ORANGE        0xFFA500
#define YELLOW        0xFFFF0A
#define YELLOW_GREEN  0xA5FF0A
#define GREEN         0x0AFF0A
#define CYAN_GREEN    0x0AFFA5
#define CYAN          0x0AFFFF
#define CYAN_BLUE     0x0AA5FF
#define BLUE          0x0A0AFF
#define PURPLE        0xA50AFF
#define VIOLET        0xFF0AFF
#define MAGENTA       0xFF0AA5

// Calibrate brightness
#define R_OFFSET  1.0
#define G_OFFSET  0.16
#define B_OFFSET  0.30

void rgb_begin();
void rgb_write(uint32_t color);
void rgb_write(uint8_t r, uint8_t g, uint8_t b);
void rgb_off();
void start_signal();

#endif // __RGB_H__

