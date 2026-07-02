// LOAD CELL RECORDING

//AUTHORS:
// Euler Xavier de Freitas
// Paulo Amaral
// Flavio Mourao (mourao.fg@gmail.com)

//Started: 04/2026
//Last update: 07/2026



#include "HX711.h"

// === PIN MAPPING ===
const int LOADCELL_DOUT_PIN = 2;
const int LOADCELL_SCK_PIN  = 3;
// HX711 board Vcc -> Arduino 5V
// HX711 board GND -> Arduino GND
// Load cell red wire   -> HX711 E+
// Load cell black wire -> HX711 E-
// Load cell white wire -> HX711 A-
// Load cell green wire -> HX711 A+

HX711 scale;

// Maximum time (ms) to wait for a new HX711 conversion before giving up
// on the current cycle. Prevents the Arduino from blocking indefinitely
// if the sensor stops responding (e.g. a loose cable).
const unsigned long READ_TIMEOUT_MS = 200;

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n=== Load Cell - Behavior Recording ===");
  Serial.println("Send 't' at any time to tare (zero) the scale.");

  scale.begin(LOADCELL_DOUT_PIN, LOADCELL_SCK_PIN);
  scale.set_gain(128);

  // === INITIAL TARE ===
  Serial.println("Performing initial tare (64 samples)...");
  scale.tare(64);                     // 64 samples = more stable zero point
  delay(200);

  long valueAfterTare = scale.get_value(10);
  Serial.print("Tare complete. Value after tare: ");
  Serial.println(valueAfterTare);
  Serial.println("If not close to zero, send 't' again.\n");
}

void loop() {
  // --- READ ---
  // A single read (scale.read()) is used per cycle instead of
  // scale.read_average(N). read_average(N) calls read() N times in a
  // row, and every call after the first BLOCKS until the HX711's next
  // conversion is ready. On this specific board the HX711 is running at
  // its high-speed ~80 SPS mode, so read_average(5) was blocking for ~4
  // conversion periods per cycle, needlessly capping the effective
  // sampling rate to a fraction of what the sensor can actually deliver.
  //
  // wait_ready_timeout() replaces the previous "if (scale.is_ready())"
  // check: in addition to checking whether a conversion is ready, it
  // gives up after READ_TIMEOUT_MS if the sensor doesn't respond, instead
  // of blocking forever.
  if (scale.wait_ready_timeout(READ_TIMEOUT_MS)) {
    long rawValue = scale.read();     // single conversion, no extra averaging
    Serial.println(rawValue);         // raw signal value only
  }

  // === TARE COMMAND ===
  if (Serial.available() > 0) {
    char c = Serial.read();
    if (c == 't' || c == 'T') {
      Serial.println("\nTaring (64 samples)...");
      scale.tare(64);                 // robust tare
      delay(200);
      long valueAfterTare = scale.get_value(10);
      Serial.print("Tare complete. Current value: ");
      Serial.println(valueAfterTare);
      Serial.println("Ready.\n");
    }
  }
}
