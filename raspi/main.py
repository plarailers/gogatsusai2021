#!/usr/bin/env python3

import subprocess
import re
import os.path
import time
import datetime
import RPi.GPIO as GPIO
import serial
import numpy as np
import queue
import threading

###パラメータ###############################
sp100 = 8.0#dc=100の時の速度[m/s]
sp50 = 4.0#dc=50の時の速度[m/s]
Kp = 0#比例ゲイン
Kd = 0#微分ゲイン
maxspeed = 0.9*sp100#想定最大速度,sp100以下
##########################################

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

r = 1.4e-2#車輪の半径[m]

timer1 = 0
timer2 = 0
#dc = 0
speed = 0
est = 0
error = [0,0]
inp = 0

def add_input(input_queue):
    while True:
        input_queue.put(input("目標値を入力してください"))

def setup():
    global process_socat, process_momo, port, inp
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
    #inp = float(input("input speed"))

def on_sensor(channel):
    global est, timer2
    data = b'o\n'
    port.write(data)
    port.flush()
    print(datetime.datetime.now(), 'send sensor', data)

    current_time = time.time()
    dt = current_time - timer2
    timer2 = current_time

    est =  2 * np.pi * r / dt #速度推定値[m/s],磁石数1

def loop():
    global speed,est,inp
    while port.in_waiting > 0:
        data = port.read()
        #speed = data[0] / 255 * maxspeed#速度目標値
        #dc = speed * 100 / 255
        #motor.ChangeDutyCycle(dc)
        print(datetime.datetime.now(), 'receive speed', speed)
    #inp = float(input())
    speed = inp / 255 * maxspeed
    dc = dc_control()
    motor.ChangeDutyCycle(dc)

    print("dc:",dc)
    print("est:",est)
    #print("speed:",speed)

    if time.time() - timer2 > 2:#ホールセンサが2秒間立ち上がらなかったらest=0とする
        est = 0 

def speed2dc():#speedの定常走行に必要なdcを求める,試験結果に基づく線形近似
    slope = 50 / (sp100 - sp50)
    intercept = - 50 * sp50 / (sp100 - sp50) + 50
    dc = slope * speed + intercept
    return dc

def dc_control():#dcを制御
    global error,timer1
    error[1] = error[0]#1ステップ前の偏差
    error[0] = speed - est#偏差

    current_time = time.time()
    dt = current_time - timer1
    timer1 = current_time

    dc = speed2dc()#フィードフォワード制御
    print("dc1",dc)
    dc += Kp * error[0] + Kd * (error[0] - error[1]) / dt#PD制御

    if dc > 100:
        dc = 100
    elif dc < 0:
        dc =0
    
    return dc

if __name__ == '__main__':
    try:
        setup()
        input_queue = queue.Queue()
        # スレッドを作成
        input_thread = threading.Thread(target=add_input, args=(input_queue,))
        input_thread.daemon = True
        input_thread.start()

        while True:
            loop()
            time.sleep(0.01)

            if not input_queue.empty():
                inp = float(input_queue.get())
                print("dc is updated to",inp)

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
