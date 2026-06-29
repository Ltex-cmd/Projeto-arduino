#include <Servo.h>

Servo spray;

void setup() {
    Serial.begin(9600);
    spray.attach(9);
    spray.write(0);
}

void loop() {

    if (Serial.available()) {

        char c = Serial.read();

        if (c == '1') {

            spray.write(90);
            delay(500);

            spray.write(0);
        }
    }
}