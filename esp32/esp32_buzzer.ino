#include <Arduino.h>

// Configure this pin according to your ESP32 board and wiring.
// Example: GPIO 18 is commonly available as a digital output.
const int BUZZER_PIN = 18;

void setup() {
  Serial.begin(9600);
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);  // Buzzer off initially
}

void loop() {
  if (Serial.available() > 0) {
    char c = Serial.read();

    if (c == '1') {
      digitalWrite(BUZZER_PIN, HIGH);  // Drowsy: buzzer ON
    } else if (c == '0') {
      digitalWrite(BUZZER_PIN, LOW);   // Awake: buzzer OFF
    } else {
      // Ignore any other characters
    }
  }
}

