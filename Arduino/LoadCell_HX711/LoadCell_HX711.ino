// LOAD CELL RECORDING

//AUTHORS:
// Euler Xavier de Freitas
// Paulo Amaral
// Flavio Mourao (mourao.fg@gmail.com)

//Started: 04/2026
//Last update: 07/2026
//  - get_value() used instead of read(), so the Tare offset is actually
//    subtracted from what's sent over serial (read() ignores it).
//  - Serial gain command added: 'h' selects Channel A gain=128 (default,
//    "high"), 'm' selects Channel A gain=64 ("medium"). Only these two
//    are offered: gain=32 belongs to Channel B, a SEPARATE physical
//    input not wired in this design (the load cell is on A+/A-) --
//    selecting it would silently read an unconnected channel, not "the
//    same signal at lower gain".


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

// Current Channel A PGA gain (128 or 64 only -- see note above).
int currentGain = 128;

void applyGain(int gain) {
  currentGain = gain;
  scale.set_gain(gain);
  // Re-tare after a gain change: the raw zero-offset is gain-dependent,
  // so a tare value taken at the previous gain no longer applies
  // correctly once the gain changes.
  Serial.print("Gain set to ");
  Serial.print(gain);
  Serial.println(". Re-taring (64 samples)...");
  scale.tare(64);
  delay(200);
  long valueAfterTare = scale.get_value(10);
  Serial.print("Re-tare complete. Value: ");
  Serial.println(valueAfterTare);
  Serial.println("Ready.\n");
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n=== Load Cell - Behavior Recording ===");
  Serial.println("Send 't' to tare, 'h' for gain 128 (default), 'm' for gain 64.");

  scale.begin(LOADCELL_DOUT_PIN, LOADCELL_SCK_PIN);
  scale.set_gain(currentGain);

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
  // A single conversion (scale.get_value(1)) is used per cycle instead of
  // scale.read_average(N) or repeated get_value(N) calls -- those block
  // for multiple HX711 conversion periods per cycle, needlessly capping
  // the effective sampling rate. get_value(1) reads once AND subtracts
  // the current tare offset (unlike scale.read(), which returns the
  // completely raw code and ignores any tare() call ever made).
  //
  // wait_ready_timeout() gives up after READ_TIMEOUT_MS if the sensor
  // doesn't respond, instead of blocking forever.
  if (scale.wait_ready_timeout(READ_TIMEOUT_MS)) {
    long rawValue = scale.get_value(1);  // single conversion, tare offset already subtracted
    Serial.println(rawValue);            // signal value only
  }

  // === SERIAL COMMANDS ===
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
    } else if (c == 'h' || c == 'H') {
      applyGain(128);
    } else if (c == 'm' || c == 'M') {
      applyGain(64);
    }
  }
}
