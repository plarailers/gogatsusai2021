"""
画像をもとに信号を検知し、信号の色を推定する。

Example:
    import cv2
    from detect_signal import detect

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    result = detect(frame)
    if not result.detected:
        pass  # 検知されなかった
    elif result.red:
        pass  # 赤が検知された
    elif result.blue:
        pass  # 青が検知された
"""

from dataclasses import dataclass
import numpy as np
import cv2

@dataclass
class DetectionResult:
    """
    検知の結果をまとめたもの。
    """
    detected: bool
    red: bool
    blue: bool


def detect(frame, debug=False):
    """
    標識を検知する。

    Args:
        frame: 検知対象の画像。
        debug: True のとき、検知結果が frame に書き加えられる。

    Returns:
        DetectionResult を返す。
    """

    result = DetectionResult(detected=False, red=False, blue=False)

    height, width, _ = frame.shape

    # 処理効率のため一部分のみ切り取る
    trim_top, trim_bottom = int(0.3 * height), int(0.8 * height)
    partial_frame = frame[trim_top:trim_bottom, :]

    # グレイスケールによる二値化
    gray = cv2.cvtColor(partial_frame, cv2.COLOR_BGR2GRAY)

    _ret, mask = cv2.threshold(gray, 32, 255, cv2.THRESH_BINARY_INV)

    # 領域抽出
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        y += trim_top # 切り取られる前の実際の位置

        # デバッグ用（矩形の表示）
        if debug:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)
 
        if (# 大きさによるフィルター
            height * 0.04 < w < height * 0.40 and
            height * 0.04 < h < height * 0.40 and
            # 縦横比によるフィルター
            0.5 <= h / w <= 1.25 and
            # 位置によるフィルター
            x >= width * 0.5
        ):
            # 処理効率のため輪郭部分についてのみ判定
            rect = frame[y : y + h, x : x + w]
            hsv = cv2.cvtColor(rect, cv2.COLOR_BGR2HSV_FULL)
            hue = hsv[:, :, 0] # 高さと変数名被っちゃったのであとできれいにする
            s = hsv[:, :, 1]
            v = hsv[:, :, 2]

            has_red = np.any(
                  (340 / 360 <= hue / 255) & (hue / 255 <= 360 / 360)
                & (0.7 <= s / 255) & (s / 255 <= 0.9)
                & (0.3 <= v / 255) & (v / 255 <= 0.5))

            has_blue = np.any(
                  (200 / 360 <= hue / 255) & (hue / 255 <= 220 / 360)
                & (0.6 <= s / 255) & (s / 255 <= 1.0)
                & (0.3 <= v / 255) & (v / 255 <= 0.5)
            )

            if has_red:
                result.detected = True
                result.red = True

            if has_blue:
                result.detected = True
                result.blue = True

            # デバッグ用（輪郭の表示）
            if debug:
                print(f"contour {w / height} {h / height}")
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
                if has_red:
                    color = (0, 0, 255)
                elif has_blue:
                    color = (255, 0, 0)
                else:
                    color = (0, 255, 0)
                cv2.drawContours(frame, [contour], 0, color, 2, offset=(0, trim_top))

            # # デバッグ用（傾きの表示）
            # if debug:
            #     vx, vy, x, y = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
            #     y += trim_top # 切り取られる前の実際の位置
            #     angle = np.arctan2(vy, vx)
            #     left_y = int((-x * vy / vx) + y)
            #     right_y = int(((width - x) * vy / vx) + y)
            #     cv2.line(frame, (width - 1, right_y), (0, left_y), color, 2)

    return result


# この中は `python3 detect_sign.py` で直接呼び出されたときのみ実行され、
# モジュールとして読み込まれたときは実行されない
def main():
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
        result = detect(frame, debug=True)
        writer.write(frame)

    writer.release()
    cap.release()
    print('done')

if __name__ == '__main__':
    main()
