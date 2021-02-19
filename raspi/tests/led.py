#!/usr/bin/env python3

import RPi.GPIO as GPIO
import time
import datetime

PIN_OUT = 14

GPIO.setmode(GPIO.BCM)
GPIO.setup(PIN_OUT, GPIO.OUT)

try:
    print("started")
    while True:
        print(datetime.datetime.now(), "GPIO", PIN_OUT, "HIGH")
        GPIO.output(PIN_OUT, GPIO.HIGH)
        time.sleep(1.0)
        print(datetime.datetime.now(), "GPIO", PIN_OUT, "LOW")
        GPIO.output(PIN_OUT, GPIO.LOW)
        time.sleep(1.0)
except KeyboardInterrupt:
    print("interrupted")
    GPIO.cleanup()
