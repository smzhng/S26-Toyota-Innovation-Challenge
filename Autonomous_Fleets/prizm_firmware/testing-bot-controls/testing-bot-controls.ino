#include <PRIZM.h>

PRIZM prizm;

char lastCmd = 'x';
unsigned long lastCommandTime = 0;
const unsigned long COMMAND_TIMEOUT = 50;  // milliseconds

void stopMotors() {
  prizm.setMotorPower(1, 0);
  prizm.setMotorPower(2, 0);
}

void setMotorsForward() {
  prizm.setMotorPower(1, 25);
  prizm.setMotorPower(2, -25);
}

void setMotorsBackward() {
  prizm.setMotorPower(1, -25);
  prizm.setMotorPower(2, 25);
}

void setMotorsLeft() {
  prizm.setMotorPower(1, 25);
  prizm.setMotorPower(2, 25);
}

void setMotorsRight() {
  prizm.setMotorPower(1, -25);
  prizm.setMotorPower(2, -25);
}


void closeGripper() {
  prizm.setServoPosition(1,35);
}

void openGripper() {
  prizm.setServoPosition(1,180);
  //delay(500);
}

void setup() {
  prizm.PrizmBegin();
  Serial.begin(9600);
  stopMotors();

  prizm.setServoSpeed(2,50);
}

void loop() {

  // Read any incoming serial command
  if (Serial.available() > 0) {
    lastCmd = Serial.read();
    lastCommandTime = millis();
  }

  // If no recent command, stop motors
  if (millis() - lastCommandTime > COMMAND_TIMEOUT) {
    stopMotors();
    return;
  }

  // Otherwise act on the most recent command
  if (lastCmd == 'w') {
    setMotorsForward();
  }
  else if (lastCmd == 's') {
    setMotorsBackward();
  }
  else if (lastCmd == 'a') {
    setMotorsLeft();
  }
  else if (lastCmd == 'd') {
    setMotorsRight();
  } else if (lastCmd == '1') {
    openGripper();
  } else if (lastCmd == '2') {
    closeGripper();
  }
  else {
    stopMotors();
  }
}