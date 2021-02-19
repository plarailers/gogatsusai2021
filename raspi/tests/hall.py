#!/usr/bin/env python3

import RPi.GPIO as GPIO
import time
import datetime

PIN_IN = 17

GPIO.setmode(GPIO.BCM)
GPIO.setup(PIN_IN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

def callback(channel):
    input_value = GPIO.input(channel)
    print(datetime.datetime.now(), "GPIO", channel, input_value)

try:
    GPIO.add_event_detect(PIN_IN, GPIO.BOTH, callback=callback, bouncetime=300)
    print("started")
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("interrupted")
    GPIO.cleanup()
