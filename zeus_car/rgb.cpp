#include "rgb.h"
#include <SoftPWM.h>

void rgb_begin() {
  for (uint8_t i = 0; i < 3; i++) {
    SoftPWMSet(RGB_PINS[i], 0);
    SoftPWMSetFadeTime(RGB_PINS[i], 100, 100);
  }
}

void rgb_write(uint32_t color) {
  uint8_t r = (color >> 16) & 0xFF;
  uint8_t g = (color >>  8) & 0xFF;
  uint8_t b = (color >>  0) & 0xFF;
  rgb_write(r, g, b);
}

void rgb_write(uint8_t r, uint8_t g, uint8_t b) {
  // calibrate brightness
  r = int(r * R_OFFSET);
  g = int(g * G_OFFSET);
  b = int(b * B_OFFSET);
  // COMMON_ANODE reverse
  #if COMMON_ANODE
    r = 255 - r;
    g = 255 - g;
    b = 255 - b;
  #endif
  // set volatge 
  SoftPWMSet(RGB_PINS[0], r);
  SoftPWMSet(RGB_PINS[1], g);
  SoftPWMSet(RGB_PINS[2], b);
}

void rgb_off() {
  rgb_write(0, 0, 0);
}

