"""
serial_comm.py

Handles serial communication with the ESP32: sending motor commands and
detecting communication timeout (consumed by fail_safe.py).

UPDATE: automatic reconnect on USB disconnect.
Previously, once the USB cable was unplugged (or the ESP32 reset/dropped
off), send_command() would just keep failing forever -- the connection
object stayed pointed at a dead port with nothing ever trying to reopen
it. Now:
  - If a write fails, the broken connection is immediately dropped.
  - Every call to send_command() checks whether we currently have a live
    connection, and if not, attempts to reopen the port automatically.
  - Reconnect attempts are throttled to once every RECONNECT_RETRY_INTERVAL_S
    seconds, so a long-unplugged cable doesn't spam the OS with open()
    calls on every single frame (this runs at ~20-30Hz from the main loop).
No changes needed in main_logic1.py / main_logic2.py / fail_safe.py --
they already call send_command() every loop, so reconnection now happens
transparently underneath them.
"""

import time
import config

try:
    import serial as pyserial
    _PYSERIAL_AVAILABLE = True
except ImportError:
    _PYSERIAL_AVAILABLE = False

RECONNECT_RETRY_INTERVAL_S = 2.0  # don't hammer the OS -- try at most every 2 seconds


class SerialComm:
    def __init__(self, port: str = config.SERIAL_PORT, baud_rate: int = config.SERIAL_BAUD_RATE):
        self.port = port
        self.baud_rate = baud_rate
        self.connection = None
        self.last_ack_time = None
        self._last_reconnect_attempt_time = None

    def open(self) -> bool:
        if not _PYSERIAL_AVAILABLE:
            print("[serial_comm.py] ERROR: pyserial not installed. "
                  "Install it with: pip install pyserial")
            return False

        try:
            self.connection = pyserial.Serial(self.port, self.baud_rate, timeout=1)
            time.sleep(2)  # allow ESP32 to reset after the serial port opens
            self.last_ack_time = time.time()
            self._last_reconnect_attempt_time = time.time()
            print(f"[serial_comm.py] Connected to ESP32 on {self.port} @ {self.baud_rate} baud")
            return True
        except Exception as e:
            print(f"[serial_comm.py] ERROR: Could not open serial port {self.port}: {e}")
            self.connection = None
            self._last_reconnect_attempt_time = time.time()
            return False

    def _close_broken_connection(self) -> None:
        """Drops a dead connection object so the next attempt starts clean."""
        if self.connection is not None:
            try:
                self.connection.close()
            except Exception:
                pass
        self.connection = None

    def _try_reconnect(self) -> bool:
        """
        Attempts to reopen the serial port, throttled so we don't hammer
        the OS with open() calls every frame while the cable is unplugged.
        Returns True if reconnected, False if not (either too soon to
        retry, pyserial missing, or the port still isn't there).
        """
        if not _PYSERIAL_AVAILABLE:
            return False

        now = time.time()
        if self._last_reconnect_attempt_time is not None:
            elapsed = now - self._last_reconnect_attempt_time
            if elapsed < RECONNECT_RETRY_INTERVAL_S:
                return False

        self._last_reconnect_attempt_time = now
        print(f"[serial_comm.py] Attempting to reconnect to {self.port}...")
        try:
            self.connection = pyserial.Serial(self.port, self.baud_rate, timeout=1)
            time.sleep(2)  # allow ESP32 to reset after the serial port opens
            self.last_ack_time = time.time()
            print(f"[serial_comm.py] Reconnected to ESP32 on {self.port} @ {self.baud_rate} baud")
            return True
        except Exception as e:
            print(f"[serial_comm.py] Reconnect attempt failed ({self.port}): {e}")
            self.connection = None
            return False

    def send_command(self, command_char: str, left_pwm: int, right_pwm: int) -> bool:
        if self.connection is None or not self.connection.is_open:
            if not self._try_reconnect():
                return False

        message = f"{command_char},{int(left_pwm)},{int(right_pwm)}\n"
        try:
            self.connection.write(message.encode("utf-8"))
            self.last_ack_time = time.time()
            return True
        except Exception as e:
            print(f"[serial_comm.py] ERROR: Failed to send command: {e}")
            # The write failed -- the port is very likely dead (cable just
            # unplugged, ESP32 reset, etc). Drop it now so the NEXT call to
            # send_command() immediately tries a fresh reconnect instead of
            # repeatedly failing writes against a broken handle.
            self._close_broken_connection()
            return False

    def is_connected(self) -> bool:
        if self.connection is None or not self.connection.is_open:
            return False
        if self.last_ack_time is None:
            return False

        elapsed_ms = (time.time() - self.last_ack_time) * 1000
        return elapsed_ms <= config.SERIAL_TIMEOUT_MS

    def close(self) -> None:
        if self.connection is not None and self.connection.is_open:
            self.connection.close()
            print("[serial_comm.py] Serial connection closed")
