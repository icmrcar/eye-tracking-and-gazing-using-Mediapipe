"""
data_logger.py

Writes one CSV row per frame/event.
"""

import csv
import os
import config

CSV_FIELDNAMES = [
    "Timestamp", "Trial_ID", "Horizontal_Ratio", "Vertical_Ratio", "EAR",
    "Eye_Direction", "Dwell_Confirmed", "Blink_Duration_ms", "Input_Source",
    "FSM_State", "Wheelchair_Command", "Left_PWM", "Right_PWM",
    "Fail_Safe_Triggered", "FPS",
]


class DataLogger:
    def __init__(self, log_path: str = config.CSV_LOG_PATH, trial_id: str = None):
        self.log_path = log_path
        self.trial_id = trial_id
        self._file = None
        self._writer = None

    def open(self) -> None:
        directory = os.path.dirname(self.log_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        file_is_new = not os.path.exists(self.log_path) or os.path.getsize(self.log_path) == 0

        self._file = open(self.log_path, mode="a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_FIELDNAMES)

        if file_is_new:
            self._writer.writeheader()
            self._file.flush()

        print(f"[data_logger.py] Logging to {self.log_path}")

    def log_frame(self, row_data: dict) -> None:
        if self._writer is None:
            return

        if self.trial_id is not None and not row_data.get("Trial_ID"):
            row_data["Trial_ID"] = self.trial_id

        safe_row = {field: row_data.get(field, "") for field in CSV_FIELDNAMES}

        self._writer.writerow(safe_row)
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.flush()
            self._file.close()
            print("[data_logger.py] Log file closed")
