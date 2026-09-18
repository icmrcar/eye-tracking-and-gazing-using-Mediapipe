/*
  esp32_firmware.ino
  Joystick (tank-mixer) + gaze/serial control, shared smoothing + brake pipeline

  ---------------------------------------------------------------------
  KEY FIX: "pull back = motors rotate right"

  An earlier "TOP-SPEED STRAIGHT LOCK" safety check was dead code:
      joyY = constrain(joyY, -200, 200);   // caps joyY at 200
      if (abs(joyY) > 200) { joyX = 0; }   // can NEVER be true after the cap above
  X-axis crosstalk from the joystick (common on cheap potentiometer
  modules, especially at full deflection) never got suppressed, and the
  differential mixer amplified it into a visible turn while reversing.
  Fixed by lowering the trigger threshold to 150 (reachable, since
  150 < the 200 cap) so the lock actually engages when pushed/pulled hard.

  Gaze/serial protocol from Python ("<F/L/R/S>,<left_pwm>,<right_pwm>\n")
  flows through the SAME smoothing + brake-release pipeline as the
  joystick. Joystick always has priority when moved.
  ---------------------------------------------------------------------

  Pins:
    LEFT MOTOR (MDDS30):  AN1 = 25   IN1 = 26
    RIGHT MOTOR (MDDS30): AN2 = 33   IN2 = 32
    JOYSTICK:              VRX = 34   VRY = 35
    BRAKE RELAY:           18

  SAFETY -- VERIFY BEFORE TRUSTING ON THE FLOOR:
  - BRAKE_ENGAGED/BRAKE_RELEASED: confirm against your actual relay wiring
    (test with wheels off the ground).
  - Confirm forward push actually drives forward on your MDDS30 wiring.
*/

#include <math.h>

// =====================================
// LEFT MOTOR (MDDS30)
// =====================================
#define AN1 25
#define IN1 26

// =====================================
// RIGHT MOTOR (MDDS30)
// =====================================
#define AN2 33
#define IN2 32

// =====================================
// JOYSTICK
// =====================================
#define VRX 34
#define VRY 35

// =====================================
// BRAKE RELAY
// =====================================
#define BRAKE_RELAY 18
#define BRAKE_ENGAGED HIGH
#define BRAKE_RELEASED LOW

// =====================================
// TUNING
// =====================================
#define JOY_DEADZONE           130   // raw mixer units, out of -200..200
#define STRAIGHT_LOCK_THRESHOLD 150  // FIXED: was 200 (unreachable), now < the 200 cap
#define BRAKE_RELEASE_DELAY    250   // ms for the physical brake to actually disengage
#define ACCEL_SMOOTHING        0.10f // higher = snappier, lower = smoother accel
#define DECEL_SMOOTHING        0.12f // rate current speed decays toward 0 when centered/stopped

#define SERIAL_BAUD_RATE   115200
#define SERIAL_TIMEOUT_MS  500

bool INVERT_Y = true;   // flip if pulling back drives forward instead of reverse
bool INVERT_X = false;  // flip if left/right feel swapped

// =====================================
// JOYSTICK CENTER (auto-calibrated in setup())
// =====================================
int centerX = 2048;
int centerY = 2048;

// =====================================
// BRAKE STATE
// =====================================
bool brakeReleased = false;
bool waitingForBrakeRelease = false;
unsigned long brakeReleaseStart = 0;

// =====================================
// MOTOR STATE (shared by joystick + gaze paths)
// =====================================
float currentLeft = 0;
float currentRight = 0;

// =====================================
// GAZE/SERIAL STATE
// =====================================
String serialBuffer = "";
unsigned long lastSerialCommandTime = 0;
long lastGazeTargetLeft = 0;
long lastGazeTargetRight = 0;
bool lastGazeWantsStop = true;

// =====================================
// Prototypes (explicit -- Arduino IDE 1.8.19's auto-prototype generation
// is known to sometimes fail on functions taking a String reference)
// =====================================
void stopMotors();
void updateMotors();
void drive(int x, int y);
void applyTargets(long targetLeft, long targetRight, bool wantsStop);
void readSerialCommands();
void parseGazeCommand(const String &line);

// =====================================
// Setup
// =====================================

void setup()
{
    Serial.begin(SERIAL_BAUD_RATE);
    serialBuffer.reserve(32);

    pinMode(IN1, OUTPUT);
    pinMode(IN2, OUTPUT);
    pinMode(BRAKE_RELAY, OUTPUT);

    ledcAttach(AN1, 5000, 8);
    ledcAttach(AN2, 5000, 8);

    stopMotors();
    digitalWrite(BRAKE_RELAY, BRAKE_ENGAGED);

    Serial.println("Calibrating joystick center...");

    long sumX = 0;
    long sumY = 0;
    for (int i = 0; i < 50; i++)
    {
        sumX += analogRead(VRX);
        sumY += analogRead(VRY);
        delay(10);
    }
    centerX = sumX / 50;
    centerY = sumY / 50;

    Serial.print("Center X=");
    Serial.println(centerX);
    Serial.print("Center Y=");
    Serial.println(centerY);

    lastSerialCommandTime = millis();
    Serial.println("READY");
}

// =====================================
// Main loop
// =====================================

void loop()
{
    int xValue = analogRead(VRX);
    int yValue = analogRead(VRY);

    int joyX = map(xValue, centerX - 1000, centerX + 1000, -200, 200);
    int joyY = map(yValue, centerY - 1000, centerY + 1000, -200, 200);

    joyX = constrain(joyX, -200, 200);
    joyY = constrain(joyY, -200, 200);

    if (INVERT_X) joyX = -joyX;
    if (INVERT_Y) joyY = -joyY;

    // 1. Deadzone -- filters hardware noise
    if (abs(joyX) < JOY_DEADZONE) joyX = 0;
    if (abs(joyY) < JOY_DEADZONE) joyY = 0;

    // 2. Top-speed straight lock -- FIXED threshold (was unreachable at
    // 200, now 150, which the value can actually exceed before hitting
    // the cap). Suppresses X-axis crosstalk when pushing/pulling hard.
    if (abs(joyY) > STRAIGHT_LOCK_THRESHOLD) {
        joyX = 0;
    }

    bool joystickActive = (abs(joyX) > 0) || (abs(joyY) > 0);

    readSerialCommands();

    if (joystickActive) {
        Serial.print("JOY X=");
        Serial.print(joyX);
        Serial.print(" Y=");
        Serial.println(joyY);
        drive(joyX, joyY);
    } else {
        // Joystick centered -- gaze/serial has control. Re-apply the last
        // known gaze target every loop (not just when a new line arrives)
        // so the smoothing ramp keeps advancing continuously, and force a
        // stop if serial has gone quiet too long (fail-safe).
        bool wantsStop = lastGazeWantsStop
            || (millis() - lastSerialCommandTime > SERIAL_TIMEOUT_MS);
        applyTargets(lastGazeTargetLeft, lastGazeTargetRight, wantsStop);
    }

    delay(20);
}

// =====================================
// Gaze/serial parsing
// =====================================

void readSerialCommands() {
    while (Serial.available() > 0) {
        char c = Serial.read();
        if (c == '\n') {
            parseGazeCommand(serialBuffer);
            serialBuffer = "";
        } else if (c != '\r' && serialBuffer.length() < 30) {
            serialBuffer += c;
        }
    }
}

void parseGazeCommand(const String &line) {
    int firstComma = line.indexOf(',');
    int secondComma = line.indexOf(',', firstComma + 1);
    if (firstComma == -1 || secondComma == -1) return;  // malformed -- ignore

    char cmd = line.charAt(0);
    int left = constrain(line.substring(firstComma + 1, secondComma).toInt(), 0, 255);
    int right = constrain(line.substring(secondComma + 1).toInt(), 0, 255);

    lastSerialCommandTime = millis();

    if (cmd == 'S' || (left == 0 && right == 0)) {
        lastGazeTargetLeft = 0;
        lastGazeTargetRight = 0;
        lastGazeWantsStop = true;
    } else {
        // F/L/R all just mean "drive at these (unsigned, forward) PWMs" --
        // gaze has no reverse gesture, direction differentiation already
        // happened Python-side via which wheel got the bigger number.
        lastGazeTargetLeft = left;
        lastGazeTargetRight = right;
        lastGazeWantsStop = false;
    }
}

// =====================================
// DRIVE (joystick tank-mixer path)
// =====================================

void drive(int x, int y)
{
    long targetLeft = y - (x * 0.6);
    long targetRight = y + (x * 0.6);

    long maxSpeed = max(abs(targetLeft), abs(targetRight));
    if (maxSpeed > 255) {
        targetLeft = (targetLeft * 255) / maxSpeed;
        targetRight = (targetRight * 255) / maxSpeed;
    }

    bool joystickCentered = (abs(x) < 30) && (abs(y) < 30);

    applyTargets(targetLeft, targetRight, joystickCentered);
}

// =====================================
// Shared smoothing + brake pipeline (joystick AND gaze paths both
// funnel through here, so behavior is consistent regardless of source)
// =====================================

void applyTargets(long targetLeft, long targetRight, bool wantsStop)
{
    if (wantsStop)
    {
        currentLeft += (0 - currentLeft) * DECEL_SMOOTHING;
        currentRight += (0 - currentRight) * DECEL_SMOOTHING;

        if (abs(currentLeft) < 5) currentLeft = 0;
        if (abs(currentRight) < 5) currentRight = 0;

        updateMotors();

        if (currentLeft == 0 && currentRight == 0)
        {
            stopMotors();
            if (brakeReleased)
            {
                Serial.println("BRAKE ENGAGED");
                digitalWrite(BRAKE_RELAY, BRAKE_ENGAGED);
                brakeReleased = false;
            }
        }
        return;
    }

    // ---- Release brake ----
    if (!brakeReleased)
    {
        Serial.println("BRAKE RELEASED");
        digitalWrite(BRAKE_RELAY, BRAKE_RELEASED);
        brakeReleased = true;
        brakeReleaseStart = millis();
        waitingForBrakeRelease = true;

        currentLeft = 0;
        currentRight = 0;
        stopMotors();
        return;
    }

    // ---- Wait for brake to physically release ----
    if (waitingForBrakeRelease)
    {
        if (millis() - brakeReleaseStart < BRAKE_RELEASE_DELAY)
        {
            stopMotors();
            return;
        }
        waitingForBrakeRelease = false;
    }

    // ---- Smooth acceleration toward target ----
    currentLeft += (targetLeft - currentLeft) * ACCEL_SMOOTHING;
    currentRight += (targetRight - currentRight) * ACCEL_SMOOTHING;

    updateMotors();
}

// =====================================
// UPDATE MOTORS
// =====================================

void updateMotors()
{
    int leftPwm = constrain((int)fabs(currentLeft), 0, 255);
    int rightPwm = constrain((int)fabs(currentRight), 0, 255);

    if (currentLeft >= 0) {
        digitalWrite(IN1, HIGH);
    } else {
        digitalWrite(IN1, LOW);
    }
    ledcWrite(AN1, leftPwm);

    if (currentRight >= 0) {
        digitalWrite(IN2, HIGH);
    } else {
        digitalWrite(IN2, LOW);
    }
    ledcWrite(AN2, rightPwm);
}

// =====================================
// STOP MOTORS
// =====================================

void stopMotors()
{
    ledcWrite(AN1, 0);
    ledcWrite(AN2, 0);
}
