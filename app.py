import os
import sys
import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk
import time
import pygame

# 리소스 파일의 경로를 찾는 함수
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller로 패키징 된 경우, _MEIPASS는 임시 디렉토리를 가리킵니다.
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# 초기 상태 저장을 위한 전역 변수 선언
initial_horizontal_change = None
initial_vertical_change = None
initial_lateral_change = None

initial_left_arm_position = None  # 왼쪽 팔 키포인트의 초기 위치를 저장할 전역 변수
address_pose_detected = False  # 어드레스 자세 감지 여부를 추적하는 변수
swing_ended = True  # 스윙 종료 여부를 추적하는 변수

# 머리 움직임 분석을 위한 전역 변수 초기화
initial_head_position = None
head_movement_detected = False

# 스윙 카운트를 위한 전역 변수 초기화
total_swings = 0
head_fixed_count = 0
head_movement_count = 0

model_path = resource_path('movenet_lighting_tflite_float16.tflite')
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)

def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            try:
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            except ValueError:
                # y1, x1, y2, x2가 유효한 숫자가 아닌 경우 무시하고 다음으로 진행
                continue

# 어깨 중간과 엉덩이 중간에 세로선 그리기 추가
def draw_midline(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    shoulder_mid = (shaped[5][:2] + shaped[6][:2]) / 2
    hip_mid = (shaped[11][:2] + shaped[12][:2]) / 2

    if (shaped[5][2] > confidence_threshold and shaped[6][2] > confidence_threshold and
            shaped[11][2] > confidence_threshold and shaped[12][2] > confidence_threshold):
        cv2.line(frame, (int(shoulder_mid[1]), int(shoulder_mid[0])), (int(hip_mid[1]), int(hip_mid[0])), (255, 0, 0), 2)

def draw_face_vertical_line(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    # 머리 상단과 목 위치를 사용하여 대략적인 이마와 턱의 위치 추정
    head_top = shaped[0][:2]  # 머리 상단(이마로 가정)
    neck = shaped[1][:2]  # 목(턱 근처로 가정)

    if (shaped[0][2] > confidence_threshold and shaped[1][2] > confidence_threshold):
        # 이마에서 턱까지의 선 그리기
        cv2.line(frame, (int(head_top[1]), int(head_top[0])), (int(neck[1]), int(neck[0])), (0, 0, 255), 2)

def is_address_pose(keypoints_with_scores, confidence_threshold=0.4):
    keypoints = np.squeeze(keypoints_with_scores)

    # 키포인트 신뢰도 체크
    if (keypoints[5][2] < confidence_threshold or keypoints[6][2] < confidence_threshold or
        keypoints[11][2] < confidence_threshold or keypoints[12][2] < confidence_threshold):
        return False  # 신뢰도가 임계값 미만인 키포인트가 있으면 어드레스 자세로 판단하지 않음

    # 어깨와 골반의 중간점 계산
    shoulder_midpoint = (keypoints[5][:2] + keypoints[6][:2]) / 2
    hip_midpoint = (keypoints[11][:2] + keypoints[12][:2]) / 2

    # 어깨와 골반의 높이 차이 계산
    vertical_diff = abs(shoulder_midpoint[1] - hip_midpoint[1])

    # 머리-목 각도와 목-골반 각도를 계산
    head_point = keypoints[0, :2]
    neck_point = keypoints[1, :2]
    hip_center = (keypoints[11, :2] + keypoints[12, :2]) / 2
    head_neck_angle = calculate_angle(head_point, neck_point)
    neck_hip_angle = calculate_angle(neck_point, hip_center)

    # 각도 차이를 계산
    angle_diff = abs(head_neck_angle - neck_hip_angle)

    # 높이 차이와 각도 차이를 기반으로 어드레스 자세와 몸통의 직선성 판별
    vertical_diff_threshold = 5  # 높이 차이 임계값
    if vertical_diff < vertical_diff_threshold and (angle_diff < 165 and angle_diff > 145):
        return True  # 어드레스 자세이며 몸통이 직선임
    else:
        return False

def calculate_angle(p1, p2):
    """두 점 p1, p2 간의 각도를 계산합니다."""
    angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) * 180.0 / np.pi
    return angle

def calculate_distance(p1, p2):
    """두 점 p1, p2 사이의 유클리드 거리를 계산합니다."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def analyze_face_pose(keypoints):
    nose = keypoints[0]  # 코
    left_ear = keypoints[3]  # 왼쪽 귀
    right_ear = keypoints[4]  # 오른쪽 귀
    neck = keypoints[1]  # 목

    # 수평 변화 감지: 귀와 귀 사이의 거리
    horizontal_change = calculate_distance(left_ear[:2], right_ear[:2])
    print(f"수평 변화 거리: {horizontal_change:.2f}")

    # 높이 변화 감지: 코와 목 사이의 거리
    vertical_change = calculate_distance(nose[:2], neck[:2])
    print(f"높이 변화 거리: {vertical_change:.2f}")

    # 좌우 거리 변화 감지: 코와 양쪽 귀의 중점 사이의 거리
    ears_midpoint = ((left_ear[0] + right_ear[0]) / 2, (left_ear[1] + right_ear[1]) / 2)
    lateral_change = calculate_distance(nose[:2], ears_midpoint)
    print(f"좌우 거리 변화: {lateral_change:.2f}")

def distance(p1, p2):
    """두 점 p1, p2 사이의 거리를 계산합니다."""
    return np.sqrt((p2[0] - p1[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def save_initial_state_face(keypoints):
    global initial_horizontal_change, initial_vertical_change, initial_lateral_change
    # 초기 상태 계산 및 저장
    nose = keypoints[0]
    left_ear = keypoints[3]
    right_ear = keypoints[4]
    neck = keypoints[1]

    initial_horizontal_change = calculate_distance(left_ear[:2], right_ear[:2])
    initial_vertical_change = calculate_distance(nose[:2], neck[:2])
    ears_midpoint = ((left_ear[0] + right_ear[0]) / 2, (left_ear[1] + right_ear[1]) / 2)
    initial_lateral_change = calculate_distance(nose[:2], ears_midpoint)

def analyze_swing_from_initial_face(keypoints):
    # 현재 상태 계산
    nose = keypoints[0]
    left_ear = keypoints[3]
    right_ear = keypoints[4]
    neck = keypoints[1]

    current_horizontal_change = calculate_distance(left_ear[:2], right_ear[:2])
    current_vertical_change = calculate_distance(nose[:2], neck[:2])
    ears_midpoint = ((left_ear[0] + right_ear[0]) / 2, (left_ear[1] + right_ear[1]) / 2)
    current_lateral_change = calculate_distance(nose[:2], ears_midpoint)

    # 초기 상태와 비교
    horizontal_movement = current_horizontal_change - initial_horizontal_change
    vertical_movement = current_vertical_change - initial_vertical_change
    lateral_movement = current_lateral_change - initial_lateral_change

    print(f"스윙 수평 변화: {horizontal_movement:.2f}, 스윙 높이 변화: {vertical_movement:.2f}, 스윙 좌우 변화: {lateral_movement:.2f}")

def save_initial_left_arm_position(keypoints):
    global initial_left_arm_position
    # 왼쪽 팔(어깨)의 키포인트 위치를 저장합니다.
    left_shoulder = keypoints[9][:2]  # 왼쪽 어깨의 키포인트 인덱스는 5번입니다.
    initial_left_arm_position = left_shoulder

# 왼쪽 어깨의 x 좌표 변화를 추적할 리스트
left_shoulder_movement_history = []
# 스윙 종료를 간주하기 위해 요구되는 연속 프레임 수
required_continuous_frames = 20

def update_left_shoulder_movement_history(current_left_shoulder_x):
    # 현재 왼쪽 어깨의 x 좌표를 이동 추적 리스트에 추가
    left_shoulder_movement_history.append(current_left_shoulder_x)
    # 리스트가 요구되는 프레임 수보다 길어지면 가장 오래된 원소 제거
    if len(left_shoulder_movement_history) > required_continuous_frames:
        left_shoulder_movement_history.pop(0)

def check_continuous_movement_to_left():
    # 이동 추적 리스트의 길이가 요구되는 프레임 수에 도달했는지 확인
    if len(left_shoulder_movement_history) == required_continuous_frames:
        # 리스트의 모든 원소가 초기 위치보다 왼쪽인지 확인
        if all(x < initial_left_shoulder_x for x in left_shoulder_movement_history):
            return True
    return False

def check_swing_end(current_left_shoulder_x):
    global swing_ended
    update_left_shoulder_movement_history(current_left_shoulder_x)

    if check_continuous_movement_to_left():
        print("스윙 종료 감지됨. 어드레스 자세 탐지로 돌아갑니다.")
        swing_ended = True
        # 스윙 종료 후 초기화 작업
        left_shoulder_movement_history.clear()
        return True
    return False

# 스윙 종료 후 어드레스 자세 탐지를 지연시키기 위한 변수
last_swing_end_time = 0  # 마지막 스윙 종료 시간
address_pose_detection_delay = 5  # 어드레스 자세 탐지를 지연시키는 시간(초)

def check_swing_and_head_movement(current_left_shoulder_x):
    global swing_ended, head_movement_detected, last_swing_end_time, total_swings, head_fixed_count, head_movement_count
    if check_swing_end(current_left_shoulder_x):
        total_swings += 1
        if head_movement_detected:
            print("스윙 종료됨, 머리 움직임")
            head_movement_count += 1
        else:
            print("스윙 종료됨, 머리 고정 성공")
            head_fixed_count += 1
        swing_ended = True
        head_movement_detected = False  # 머리 움직임 감지 변수 초기화
        last_swing_end_time = time.time()  # 스윙 종료 시간 기록

def analyze_head_movement_during_swing(keypoints):
    global head_movement_detected, initial_head_position
    # 현재 머리 위치를 계산
    current_head_position = keypoints[0][:2]  # 코 키포인트의 x, y 좌표

    # 초기 상태와 현재 상태 사이의 거리 차이 계산
    movement_distance = calculate_distance(initial_head_position, current_head_position)

    # 특정 임계값 이상 움직였는지 판별
    if movement_distance > 100:  # 임계값 설정 필요
        head_movement_detected = True

def save_initial_head_position(keypoints):
    global initial_head_position
    # 머리 위치(코 키포인트)를 초기 상태로 저장
    nose_keypoint = keypoints[0][:2]  # 코 키포인트의 x, y 좌표
    initial_head_position = nose_keypoint

EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        # Pygame 초기화 및 사운드 파일 로드
        pygame.init()
        self.success_sound = pygame.mixer.Sound(resource_path('Golfswing_success_sound.mp3'))
        self.fail_sound = pygame.mixer.Sound(resource_path('Golfswing_Fail_sound.mp3'))

        # 아이콘 이미지 로드
        icon_image = tk.PhotoImage(file=resource_path('handy_caddy_icon.png'))
        self.window.iconphoto(False, icon_image)

        # 메인 프레임 생성
        main_frame = ttk.Frame(window)
        main_frame.grid(row=0, column=0, padx=10, pady=10)

        # 스타일 설정
        style = ttk.Style()
        style.configure('My.TFrame', background='white')
        style.configure('My.TLabel', background='white', foreground='black')

        # 타이틀
        self.title = ttk.Label(main_frame, text="Handy Caddy", font=("Helvetica", 16))
        self.title.grid(row=0, column=0, columnspan=3)

        # 숫자 라벨의 폭 기준 설정
        num_width = 8

        self.total_swings_label = tk.Label(main_frame, text="Total", font=("Arial", 20), background='white', foreground='black', width=num_width, anchor='center')
        self.total_swings_label.grid(row=1, column=0, pady=(5, 0), sticky='n')
        self.total_swings_count = tk.Label(main_frame, text="0", font=("Arial", 20), background='white', foreground='black', width=num_width, height=2, anchor='center')
        self.total_swings_count.grid(row=2, column=0, sticky='n')

        self.head_fixed_label = tk.Label(main_frame, text="Success", font=("Arial", 20), background='white', foreground='black', width=num_width, anchor='center')
        self.head_fixed_label.grid(row=1, column=1, pady=(5, 0), sticky='n')
        self.head_fixed_count_label = tk.Label(main_frame, text="0", font=("Arial", 20), background='white', foreground='black', width=num_width, height=2, anchor='center')
        self.head_fixed_count_label.grid(row=2, column=1, sticky='n')

        self.head_movement_label = tk.Label(main_frame, text="Fail", font=("Arial", 20), background='white', foreground='black', width=num_width, anchor='center')
        self.head_movement_label.grid(row=1, column=2, pady=(5, 0), sticky='n')
        self.head_movement_count_label = tk.Label(main_frame, text="0", font=("Arial", 20), background='white', foreground='black', width=num_width, height=2, anchor='center')
        self.head_movement_count_label.grid(row=2, column=2, sticky='n')

        # 비디오 소스 열기
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # 위의 비디오 소스 크기에 맞는 캔버스 생성
        self.canvas = tk.Canvas(main_frame, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.grid(row=3, column=0, columnspan=3, pady=(10,0))

        # 버튼을 위한 프레임 생성
        button_frame = ttk.Frame(main_frame, style='My.TFrame')
        button_frame.grid(row=4, column=0, columnspan=3, sticky='w')

        # 버튼들 생성 및 배치
        self.btn_start = ttk.Button(button_frame, text="Start", command=self.start_video)
        self.btn_start.grid(row=0, column=0, padx=(0, 5))
        self.btn_stop = ttk.Button(button_frame, text="Stop", command=self.stop_video)
        self.btn_stop.grid(row=0, column=1, padx=(0, 5))
        self.btn_reset = ttk.Button(button_frame, text="Reset", command=self.reset_counters)
        self.btn_reset.grid(row=0, column=2, padx=(0, 5))

        self.delay = 15
        self.update()
        self.window.mainloop()

    def start_video(self):
        self.vid = cv2.VideoCapture(self.video_source)

    def stop_video(self):
        self.vid.release()

    def reset_counters(self):
        global total_swings, head_fixed_count, head_movement_count
        total_swings = 0
        head_fixed_count = 0
        head_movement_count = 0
        self.update_labels_with_color(fixed=False, movement=False)

    def reset_label_colors(self):
        """라벨의 색상을 원래 상태로 복원하고 숫자를 표시합니다."""
        self.head_fixed_count_label.config(background='white', foreground='black', text=f"{head_fixed_count}")
        self.head_movement_count_label.config(background='white', foreground='black', text=f"{head_movement_count}")

    def update_labels_with_color(self, fixed, movement):
        """라벨의 숫자와 색상을 업데이트합니다."""
        self.total_swings_count.config(text=f"{total_swings}")
        self.head_fixed_count_label.config(text=f"{head_fixed_count}")
        self.head_movement_count_label.config(text=f"{head_movement_count}")

        # 색상 및 텍스트 변경 + 소리 재생
        if fixed:
            self.head_fixed_count_label.config(background='green', foreground='white', text="O")
            pygame.mixer.Sound.play(self.success_sound)  # 성공 사운드 재생
        if movement:
            self.head_movement_count_label.config(background='red', foreground='white', text="X")
            pygame.mixer.Sound.play(self.fail_sound)  # 실패 사운드 재생

        # 2초 후 색상 복원 및 숫자 표시
        self.window.after(2000, self.reset_label_colors)

    def update(self):
        global swing_ended, last_swing_end_time, address_pose_detection_delay, initial_left_shoulder_x, head_movement_detected, total_swings, head_fixed_count, head_movement_count
        ret, frame = self.vid.read()
        if ret:
            # 모델에 프레임 전달 및 추론
            img = frame.copy()
            img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
            input_image = tf.cast(img, dtype=tf.uint8)
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
            interpreter.invoke()
            keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

            keypoints = np.squeeze(np.multiply(keypoints_with_scores, [frame.shape[0], frame.shape[1], 1]))
            keypoints = keypoints[:, :2]

            draw_connections(frame, keypoints_with_scores, EDGES, 0.4)
            draw_keypoints(frame, keypoints_with_scores, 0.4)
            draw_midline(frame, keypoints_with_scores, 0.4)

            current_time = time.time()
            if swing_ended and (current_time - last_swing_end_time) > address_pose_detection_delay:
                if is_address_pose(keypoints_with_scores, 0.5):
                    print("올바른 어드레스 자세 감지됨. 초기 상태를 저장합니다.")
                    initial_left_shoulder_x = keypoints[9][0]
                    save_initial_head_position(keypoints)
                    address_pose_detected = True
                    swing_ended = False
                    left_shoulder_movement_history.clear()
                    head_movement_detected = False
            elif not swing_ended:
                current_left_shoulder_x = keypoints[9][0]
                analyze_head_movement_during_swing(keypoints)
                check_swing_and_head_movement(current_left_shoulder_x)

                # 카운터 및 색상 업데이트
                self.update_labels_with_color(not head_movement_detected, head_movement_detected)

            # OpenCV 이미지 -> PIL 포맷으로 변환 -> Tkinter에 표시
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

# Tkinter 윈도우 생성 및 애플리케이션 실행
root = tk.Tk()
app = App(root, "Handy Caddy")
root.mainloop()
