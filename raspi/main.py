#!/usr/bin/env python3

import subprocess
import re
import os.path
import time
import datetime
import RPi.GPIO as GPIO
import serial

MOTOR_PIN = 19
SENSOR_PIN = 10

GPIO.setmode(GPIO.BCM)
GPIO.setup(MOTOR_PIN, GPIO.OUT)
GPIO.setup(SENSOR_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

motor = GPIO.PWM(MOTOR_PIN, 50)

MOMO_BIN = os.path.expanduser('~/momo-2020.8.1_raspberry-pi-os_armv6/momo')

process_socat = None
process_momo = None
port = None

def setup():
    global process_socat, process_momo, port
    print('starting...')
    process_socat = subprocess.Popen(['socat', '-d', '-d', 'pty,raw,echo=0', 'pty,raw,echo=0'], stderr=subprocess.PIPE)
    port1_name = re.search(r'N PTY is (\S+)', process_socat.stderr.readline().decode()).group(1)
    port2_name = re.search(r'N PTY is (\S+)', process_socat.stderr.readline().decode()).group(1)
    process_socat.stderr.readline()
    print('using ports', port1_name, 'and', port2_name)
    process_momo = subprocess.Popen([MOMO_BIN, '--no-audio-device', '--use-native', '--force-i420', '--serial', f'{port1_name},9600', 'test'])
    port = serial.Serial(port2_name, 9600)
    motor.start(0)
    GPIO.add_event_detect(SENSOR_PIN, GPIO.RISING, callback=on_sensor, bouncetime=10)
    print('started')
    print('motor:', MOTOR_PIN)
    print('sensor:', SENSOR_PIN)
    print('running at http://raspberrypi.local:8080/')
    print('Ctrl+C to quit')

def on_sensor(channel):
    data = b'o\n'
    port.write(data)
    port.flush()
    print(datetime.datetime.now(), 'send sensor', data)

def loop():
    while port.in_waiting > 0:
        data = port.read()
        speed = data[0]
        dc = speed * 100 / 255
        motor.ChangeDutyCycle(dc)
        print(datetime.datetime.now(), 'receive speed', speed)

if __name__ == '__main__':
    try:
        setup()
        while True:
            loop()
            time.sleep(0.01)
    except KeyboardInterrupt:
        print('interrupted')
    except Exception as e:
        print(e)
    finally:
        motor.stop()
        GPIO.cleanup()
        if port:
            port.close()
        if process_momo:
            process_momo.terminate()
        if process_socat:
            process_socat.terminate()
