"""Sends S,steer,throttle to the ESP32.

There is deliberately no background sender. Commands only go out from the main
loop, so if that loop hangs or crashes the ESP32's 300 ms watchdog stops the car.
"""
import time

import serial

import config as cfg


class EspLink:
    def __init__(self, port=None, baud=None, ready_timeout=6.0):
        self.ser = serial.Serial(port or cfg.ESP_PORT, baud or cfg.ESP_BAUD,
                                 timeout=0.1, write_timeout=0.1)
        self._wait_ready(ready_timeout)

    def _wait_ready(self, timeout):
        # Opening the port usually resets the ESP32, which then spends 3 s arming.
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if b"READY" in self.ser.readline():
                return
        print("WARNING: no READY from the ESP32. Continuing at neutral.")
        self.send(0, 0)

    def send(self, steer, throttle):
        steer = max(-1.0, min(1.0, steer + cfg.STEER_TRIM))
        throttle = max(-1.0, min(1.0, throttle))
        self.ser.write(f"S,{steer:.3f},{throttle:.3f}\n".encode())
        if self.ser.in_waiting:
            self.ser.reset_input_buffer()   # discard watchdog status messages

    def close(self):
        try:
            for _ in range(3):
                self.send(0, 0)
                time.sleep(0.02)
        finally:
            self.ser.close()
