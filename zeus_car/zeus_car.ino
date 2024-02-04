#include <Arduino.h>
#include <SoftPWM.h>
#include <string.h>

/*
 *  [0]--|||--[1]
 *   |         |
 *   |         |
 *   |         |
 *   |         |
 *  [3]-------[2]
 */

/** Set the pins for the motors */
#define MOTOR_PINS \
    (uint8_t[8]) { \
        3, 4, 5, 6, A3, A2, A1, A0 \
    }

/** Set the positive and negative directions for the motors */
#define MOTOR_DIRECTIONS \
    (uint8_t[4]) { \
        1, 0, 0, 1 \
    }

#define MOTOR_POWER_MIN 28  // TODO: double check
#define MOTOR_POWER_MAX 255 // TODO: double check

/** Configure Wifi mode, SSID, password*/
#define WIFI_MODE WIFI_MODE_AP
#define SSID "Zeus_Car"
#define PASSWORD "12345678"

// Globals
Esp32Listener esp_listener = Esp32Listener();

void setup() {
    Serial.begin(115200);
    Serial.println("Initializing zeus_car");
    SoftPWMBegin();  
    rgb_begin();
    rgb_write(ORANGE);
    car_begin();      
    esp_listener.begin(SSID, PASSWORD, WIFI_MODE, PORT);
    esp_listener.set_on_receive(on_receive);
}

// Program loop
void loop() {
    
    mode, action = esp_listener.listen();

    switch (mode) {

        case STANDBY:
            rgb_write(PURPLE);
            car_stop();
            delay(1000);
            break;

        case ACT:
            rgb_write(GREEN);
            car_move(action);
            break;
    }
}


void start_car() {
    for (uint8_t i = 0; i < 8; i++) {
        SoftPWMSet(MOTOR_PINS[i], 0);
        SoftPWMSetFadeTime(MOTOR_PINS[i], 100, 100);
    }
}

void set_motors(int8_t power0, int8_t power1, int8_t power2, int8_t power3) {
    bool dir[4];
    int8_t power[4] = { power0, power1, power2, power3 };
    int8_t newPower[4];

    for (uint8_t i = 0; i < 4; i++) {
        dir[i] = power[i] > 0;

        // TODO: why invert it?
        if (MOTOR_DIRECTIONS[i]) { 
            dir[i] = !dir[i];
        }

        if (power[i] == 0) {
            newPower[i] = 0;
        } else {
            newPower[i] = map(abs(power[i]), 0, 100, MOTOR_POWER_MIN, MOTOR_POWER_MAX);
        }
        SoftPWMSet(MOTOR_PINS[i * 2], dir[i] * newPower[i]);
        SoftPWMSet(MOTOR_PINS[i * 2 + 1], !dir[i] * newPower[i]);
    }
}

// TODO: make standby functionality
void standby() {

} 

// TODO: make action functionality
void act() {
    rgbWrite(MODE_APP_CONTROL_COLOR);
    carMoveFieldCentric(remoteAngle, remotePower, remoteHeading, appRemoteDriftEnable);
    lastRemotePower = remotePower;
} 

// websocket received data processing
void on_receive() {
  // Serial.print("recv:");Serial.println(aiCam.recvBuffer);
  irOrAppFlag = true;

  // Mode select: line track without magnetic field, line track withmagnetic field, obstacle following, obstacle avoidance
  current_button_state[0] = aiCam.getSwitch(REGION_M);
  current_button_state[1] = aiCam.getSwitch(REGION_N);
  current_button_state[2] = aiCam.getSwitch(REGION_O);
  current_button_state[3] = aiCam.getSwitch(REGION_P);

  // check change
  bool is_change = false;
  for(int i = 0; i < 4; i++) {
    if(current_button_state[i] != last_button_state[i]) {
      is_change = true;
      last_button_state[i] = current_button_state[i];
    }
  }
  // changed
  if (is_change || currentMode == MODE_APP_CONTROL) {
    if (current_button_state[0]) {
        if(currentMode != MODE_LINE_TRACK_WITHOUT_MAG) {
          carResetHeading();
          currentMode = MODE_LINE_TRACK_WITHOUT_MAG;
        }
    } else if (current_button_state[1]) {
      if(currentMode != MODE_LINE_TRACK_WITH_MAG) {
        carResetHeading();
        currentMode = MODE_LINE_TRACK_WITH_MAG;
      }
    } else if (current_button_state[2]) {
      if(currentMode != MODE_OBSTACLE_FOLLOWING) {
        carResetHeading();
        currentMode = MODE_OBSTACLE_FOLLOWING;
      }
    } else if (current_button_state[3]) {
      if(currentMode != MODE_OBSTACLE_AVOIDANCE) {
        carResetHeading();
        currentMode = MODE_OBSTACLE_AVOIDANCE;
      }
    } else {
      if(currentMode != MODE_APP_CONTROL) {
        appRemoteHeading = 0;
        currentMode = MODE_NONE;
      }
    }
  }

  // Stop
  if (aiCam.getButton(REGION_F)) {
    currentMode = MODE_NONE;
    stop();
    return;
  }

  // Compass Calibrate
  if (aiCam.getButton(REGION_E)) {
    currentMode = MODE_COMPASS_CALIBRATION;
    carMove(0, 0, CAR_CALIBRATION_POWER); // rote to calibrate
    compassCalibrateStart();
    return;
  }

  // Reset Origin
  if (aiCam.getButton(REGION_G)) {
    currentMode = MODE_APP_CONTROL;
    carStop();
    carResetHeading();
    appRemoteHeading = 0;
    remoteHeading = 0;
    return;
  }

  //Joystick
  uint16_t angle = aiCam.getJoystick(REGION_K, JOYSTICK_ANGLE);
  uint8_t power = aiCam.getJoystick(REGION_K, JOYSTICK_RADIUS);
  // power = map(power, 0, 100, 0, CAR_DEFAULT_POWER);

  if ( appRemoteAngle != angle || appRemotePower != power || angle != 0 || power != 0) {
    if (currentMode != MODE_APP_CONTROL) {
      currentMode = MODE_APP_CONTROL;
      carResetHeading();
    }
    appRemoteAngle = angle;
    remoteAngle = appRemoteAngle;
    appRemotePower = power;
    remotePower = appRemotePower;
    appRemoteHeading = 0;
    remoteHeading = 0; // reset remoteHeading parameter, avoid "IR remote control" changed this value
  }

  // Drift 
  if (appRemoteDriftEnable != aiCam.getSwitch(REGION_J)) {
    if (currentMode != MODE_APP_CONTROL) {
      currentMode = MODE_APP_CONTROL;
      carResetHeading();
    }
    appRemoteDriftEnable = !appRemoteDriftEnable;
  }

  // MoveHead
  int moveHeadingA = aiCam.getJoystick(REGION_Q, JOYSTICK_ANGLE);
  int16_t moveHeadingR = aiCam.getJoystick(REGION_Q, JOYSTICK_RADIUS);
  if (appRemoteHeading != 0 || appRemoteHeadingR != 0 || appRemoteHeading != moveHeadingA || appRemoteHeadingR !=  moveHeadingR){
    if (currentMode != MODE_APP_CONTROL) {
      currentMode = MODE_APP_CONTROL;
      carResetHeading();
    }
    appRemoteAngle = angle;
    remoteAngle = appRemoteAngle;
    appRemotePower = power;
    remotePower = appRemotePower;
    appRemoteHeading = moveHeadingA;
    appRemoteHeadingR = moveHeadingR;
    remoteHeading = moveHeadingA;
    if (appRemoteDriftEnable && moveHeadingR == 0) { // Drift mode
      carResetHeading();
      appRemoteHeading = 0;
      remoteHeading = 0;
    }
  }

  // Speech control
  char speech_buf_temp[20];

  aiCam.getSpeech(REGION_I, speech_buf_temp);
  if (strlen(speech_buf_temp) > 0) {
    if (aiCam.send_doc["I"].isNull() == false) {
      bool _last_stat = aiCam.send_doc["I"].as<bool>();
      if (_last_stat == 1) {
        aiCam.send_doc["I"] = 0;
      } else {
        aiCam.send_doc["I"] = 1;
      }
    } else {
      aiCam.send_doc["I"] = 0;
    }
  } 

  if (strcmp(speech_buf_temp, speech_buf) != 0) {
    strcpy(speech_buf, speech_buf_temp);
    if (strlen(speech_buf) > 0) {
      int8_t cmd_code = text_2_cmd_code(speech_buf);
      if (cmd_code != -1) {
        remotePower = SPEECH_REMOTE_POWER;
        currentMode = MODE_APP_CONTROL;
        cmd_fuc_table[cmd_code]();
      }
    }
  }

}
