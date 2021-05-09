#!/usr/bin/env python3
FPS = 30
W = 640
BUF = 1
NORMAL_INP = 15
slow_inp = 10
stop_dist = 400
slow_dist = 800
cruise_inp = 20
stop_time = 3

import time
import RPi.GPIO as GPIO
import numpy as np
import queue
import threading
import cv2
#from detect_sign import detect
#from detect_signal import detect
import detect_sign 
import detect_signal

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

r = 1.4e-2#車輪の半径[m]

timer1 = 0
timer2 = 0
newtime = 0
oldtime = 0
#dc = 0
speed = 0
est = 0
error = [0,0]
inp = 0

def add_input(input_queue):
    while True:
        input_queue.put(input("目標値を入力してください"))

def setup():
    global inp
    print('starting...')
    motor.start(0)
    GPIO.add_event_detect(SENSOR_PIN, GPIO.RISING, callback=on_sensor, bouncetime=10)
    print('started')
    print('motor:', MOTOR_PIN)
    print('sensor:', SENSOR_PIN)
    print('running at http://raspberrypi.local:8080/')
    print('Ctrl+C to quit')

def on_sensor(channel):#磁石を検知したときに呼び出される
    global est, timer2
    current_time = time.time()
    dt = current_time - timer2
    timer2 = current_time

    est =  2 * np.pi * r / dt #速度推定値[m/s],磁石数1

def loop():
    global speed,est,inp
    speed = inp * maxspeed / 100#速度目標値[m/s]
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
    dc += Kp * error[0] + Kd * (error[0] - error[1]) / dt#PD制御

    if dc > 100:
        dc = 100
    elif dc < 0:
        dc =0
    
    return dc

if __name__ == '__main__':
    input_path = 0
    output_path = 'movie.mp4'

    cap = cv2.VideoCapture(input_path)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, W*3/4)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, BUF)
    print("fps",fps)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    try:
        setup()
        input_queue = queue.Queue()
        # スレッドを作成
        input_thread = threading.Thread(target=add_input, args=(input_queue,))
        input_thread.daemon = True
        input_thread.start()

        while True:
            oldtime = newtime
            newtime = time.time()
            print("interval:", newtime - oldtime)
            loop()
            time.sleep(0.01)

            ret, frame = cap.read()
            if not ret:
                break
            result = detect_signal.detect(frame, debug=True)
            print("Red is "+str(result.red))
            dist = detect_sign.detect(frame, inp, debug=True)
            print(dist)
            
            writer.write(frame)
	    
            if dist is not None and dist < stop_dist:
                 if result.red:
                     inp = 0.0
                 elif result.blue:
                     inp = cruise_inp
            elif dist is not None and dist < slow_dist:
                 inp = slow_inp
            elif est < 0.1:
                 time.sleep(stop_time)
                 inp = cruise_inp
                 time.sleep(2)
            if not input_queue.empty():
                inp = float(input_queue.get())#speed = inp/maxspeed
                print("inp is updated to",inp)

    except KeyboardInterrupt:
        print('interrupted')
    except Exception as e:
        print(e)
    finally:
        motor.stop()
        GPIO.cleanup()

        writer.release()
        cap.release()
        print('done')
