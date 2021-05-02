#速度測定用プログラム
import time
import RPi.GPIO as GPIO
import numpy as np

MOTOR_PIN = 19
SENSOR_PIN = 10

GPIO.setmode(GPIO.BCM)
GPIO.setup(MOTOR_PIN, GPIO.OUT)
GPIO.setup(SENSOR_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

motor = GPIO.PWM(MOTOR_PIN, 50)

r = 1.4e-2#車輪の半径[m]

timer2 = 0
est = 0

def setup():
    GPIO.add_event_detect(SENSOR_PIN, GPIO.RISING, callback=on_sensor, bouncetime=50)
    motor.start(0)

def on_sensor(channel):
    global est, timer2
    
    current_time = time.time()
    dt = current_time - timer2
    timer2 = current_time

    est =  2 * np.pi * r / dt
    print("ホールセンサ読んだよ!!!!!")

if __name__ == '__main__':
    try:
        setup()
        dc = int(input("Input dc:"))
        motor.ChangeDutyCycle(dc)
        while True:
            #motor.ChangeDutyCycle(dc)
            print("Estimated speed is",est,"when dc is",dc)
            #print("ホールセンサ読んだよ!!!!!")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print('interrupted')
    except Exception as e:
        print(e)
    finally:
        motor.stop()
        GPIO.cleanup()

    
