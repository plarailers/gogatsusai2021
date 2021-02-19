#!/usr/bin/env python3

import RPi.GPIO as GPIO
import time
import datetime

PIN_OUT = 14

GPIO.setmode(GPIO.BCM)
GPIO.setup(PIN_OUT, GPIO.OUT)

p = GPIO.PWM(PIN_OUT, 50)
p.start(0)

try:
    print("started")
    while True:
        for dc in range(0, 100, 5):
            p.ChangeDutyCycle(dc)
            time.sleep(0.05)
        for dc in range(100, 0, -5):
            p.ChangeDutyCycle(dc)
            time.sleep(0.05)
except KeyboardInterrupt:
    print("interrupted")
    GPIO.cleanup()
