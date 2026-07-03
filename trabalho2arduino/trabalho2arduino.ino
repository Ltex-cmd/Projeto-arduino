#include <Servo.h>

Servo spray;

const unsigned long COOLDOWN_MS = 4000;  // trava de seguranca no proprio firmware
unsigned long lastFire = 0;

void setup() {
    Serial.begin(9600);
    spray.attach(9);
    spray.write(0);   // braco recolhido no boot
}

void loop() {

    if (Serial.available()) {

        char c = Serial.read();

        if (c == '1') {
            if (millis() - lastFire > COOLDOWN_MS) {
                lastFire = millis();

                spray.write(90);
                delay(500);

                spray.write(0);
                Serial.println("FIRED");     // ACK pro Python
            } else {
                Serial.println("COOLDOWN");  // recebeu mas segurou o jato
            }
        }
    }
}
