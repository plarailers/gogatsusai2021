"""
画像をもとに標識を検知し、標識までの推定距離を算出する。

Example:
    import cv2
    from detect_sign import detect

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    dist = detect(frame)
"""

import numpy as np
import cv2

def extract(frame):
    """mask画像を返す。"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV_FULL)
    mask = np.zeros((hsv.shape[0], hsv.shape[1], 1), dtype=np.uint8)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    mask[
         ((h > 150) | (h < 30))
        & (s >= 64)
        & (v >= 70)
        ] = 255
    return mask

    mask[
        (h < 250 / 2) | (h > 350 / 2)
        | (s < 70) | (s > 170)
        | (v < 110) | (v > 210)
        ] = 255
    return mask


def diagonal_height_ratio(h, height):
    """画面の高さに対する赤い四角形の高さを返す。"""
    return h/height


def distance(h, height):
    """標識までの距離を計算する。"""
    a = diagonal_height_ratio(h, height)
    return 29.4/a


def detect(frame, debug=False):
    """
    標識を検知する。

    Args:
        frame: 検知対象の画像。
        debug: True のとき、検知結果が frame に書き加えられる。

    Returns:
        標識が検知されたとき、標識までの距離を数値 (mm) で返す。
        検知されなかったとき、None を返す。
    """

    height, width, _ = frame.shape
    mask = extract(frame)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected = False
    detected_contours = []

    # 画面に占める枠の割合が縦横共に0.04以上なら
    w_max, h_max = -1, -1
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        _, minWH, angle = cv2.minAreaRect(contour) # 最小の四角形
        minW, minH = minWH
        if w > height * 0.04 and h > height * 0.04 and 35 <= angle <= 55 and 0.6*w < minW < 0.9*w and 0.6*h < minH < 0.9*h:
            detected = True
            detected_contours.append(contour)
            if debug:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            w_max, h_max = max(w, w_max), max(h, h_max)

    if detected:
        dist = distance(h_max, height)
    else:
        dist = None

    if debug:
        cv2.drawContours(frame, detected_contours, -1, (0, 255, 0), 3)
        putText(frame, w_max, h_max, dist)

    return dist


def putText(frame, w, h, dist):
    """赤い四角形のサイズや標識までの距離を動画に入れる。"""
    if (w < 0 or h < 0):
        cv2.putText(frame, "red rect size: - * -", (50, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 5, cv2.LINE_AA)
        cv2.putText(frame, "distance: -mm", (50, 150), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 5, cv2.LINE_AA)
    else:
        cv2.putText(frame, f"red rect size: {w} * {h}", (50, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 5, cv2.LINE_AA)
        cv2.putText(frame, f"distance: {round(dist)}mm", (50, 150), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 5, cv2.LINE_AA)


# この中は `python3 detect_sign.py` で直接呼び出されたときのみ実行され、
# モジュールとして読み込まれたときは実行されない
if __name__ == '__main__':
    input_path = 'path/to/input/movie' #入力動画ファイルへのpathを入れる
    output_path = 'path/to/output/movie' #出力動画ファイルへのpathに変更すること

    cap = cv2.VideoCapture(input_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        dist = detect(frame, debug=True)
        writer.write(frame)

    writer.release()
    cap.release()
    print('done')
