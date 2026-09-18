"""
joystick_input.py

Reads joystick axis and button state (USB gamepad via pygame), applies
deadzone filtering, and detects the emergency-stop button press. Does not
decide control authority itself -- that is input_arbiter.py's job.
"""

import config

try:
    import pygame
    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False


class JoystickCommand:
    def __init__(self, x: float, y: float, in_deadzone: bool, emergency_pressed: bool):
        self.x = x
        self.y = y
        self.in_deadzone = in_deadzone
        self.emergency_pressed = emergency_pressed


class JoystickInput:
    def __init__(self):
        self.device = None

    def initialize(self) -> bool:
        if not _PYGAME_AVAILABLE:
            print("[joystick_input.py] WARNING: pygame not installed. "
                  "Running gaze-only, no manual override / e-stop available.")
            return False

        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            print("[joystick_input.py] WARNING: No joystick detected. "
                  "Running gaze-only, no manual override / e-stop available.")
            return False

        self.device = pygame.joystick.Joystick(0)
        self.device.init()
        print(f"[joystick_input.py] Joystick initialized: {self.device.get_name()}")
        return True

    def read_state(self) -> JoystickCommand:
        if self.device is None:
            return JoystickCommand(x=0.0, y=0.0, in_deadzone=True, emergency_pressed=False)

        pygame.event.pump()

        x = self.device.get_axis(0)
        y = self.device.get_axis(1)

        in_deadzone = (abs(x) < config.JOYSTICK_DEADZONE and abs(y) < config.JOYSTICK_DEADZONE)

        emergency_pressed = False
        if self.device.get_numbuttons() > config.JOYSTICK_EMERGENCY_BUTTON_INDEX:
            emergency_pressed = bool(self.device.get_button(config.JOYSTICK_EMERGENCY_BUTTON_INDEX))

        return JoystickCommand(x=x, y=y, in_deadzone=in_deadzone, emergency_pressed=emergency_pressed)
