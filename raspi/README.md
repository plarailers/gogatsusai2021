# raspi

## メインプログラム

- main.py
  - Raspberry Pi 上で動かすスクリプト。ビデオの送信、メッセージの送受信、GPIO の操作を行う。
- index.html
  - Raspberry Pi で配信し、PC のブラウザ上で動かす。ビデオの受信、メッセージの送受信を行う。

### Raspberry Pi 側で必要なライブラリ

```
sudo apt install socat
pip3 install pyserial
```

### PC から Raspberry Pi へのファイル転送

```
scp -r ./raspi pi@raspberrypi.local:~/
```

### 実行

```
cd ~/raspi
python3 main.py
```

http://raspberrypi.local:8080/ にアクセスする。

## 動作確認用プログラム

- led.py
  - Lチカ（点滅）
  - GPIO 14番ピンで出力
- pwm.py
  - Lチカ（PWM）
  - GPIO 14番ピンで出力
- hall.py
  - センサ読み取り
  - GPIO 17番ピンで入力
