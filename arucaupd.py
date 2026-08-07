#!/usr/bin/env python3

import math
import socket
import time
import cv2
import numpy as np
import threading
from flask import Flask, Response, render_template_string, request, jsonify

CAMERA_INDEX = 0
ROBOT_MARKER_ID = 0  
ROBOT_ADDRESS = ("192.168.4.1", 8888)

INVERT_ROBOT_HEADING = False 

LINEAR_SPEED_MM_S = 320
ANGULAR_SPEED_MRAD_S = 3500

ENEMY_MARKER_IDS = {1,2, 6, 7}
ATTACK_SPEED_MM_S = 400
STEERING_KP = 2.5

TARGET_TOLERANCE_PX = 70
ANGLE_TOLERANCE_RAD = math.radians(10)
SEND_PERIOD_SECONDS = 0.05

MARKER_TIMEOUT_SEC = 0.5  

SERVO_CENTER_DEG = 90
SERVO_MAX_DEG = 160
SERVO_MIN_DEG = 20
SERVO_PAUSE_SEC = 0.5

MAP_SIZE = (600, 600)  
GRID_STEP = 50         

target_pixel: tuple[int, int] | None = None
target_marker_id: int | None = None  
visited_marker_ids: set[int] = set() 
path_history: list[tuple[int, int]] = []

servo_active: bool = True
servo_step: int = 0
servo_step_time: float = 0.0
servo_pause_current: float = 0.5

last_robot_center: np.ndarray | None = None
last_robot_heading: np.ndarray | None = None
last_robot_see_time: float = 0.0

latest_jpeg_frame: bytes | None = None
frame_width: int = 640
frame_height: int = 480
state_str: str = "SEARCHING TARGETS"
current_command: str = "STOP"
autopilot_enabled: bool = True
waiting_at_target: bool = False
target_arrival_time: float = 0.0
hunt_mode_enabled: bool = True

left_line_sensor: int = 0
right_line_sensor: int = 0
avoid_state: str = "NORMAL"
avoid_end_time: float = 0.0
avoid_turn_dir: int = 1
LINE_THRESHOLD: int = 600


app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>NEYMARK Robot Control</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #121214;
            color: #e1e1e6;
            margin: 0;
            padding: 15px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .container {
            width: 100%;
            max-width: 640px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        h1 {
            font-size: 1.5rem;
            margin: 10px 0;
            color: #00f0ff;
            text-shadow: 0 0 10px rgba(0, 240, 255, 0.3);
        }
        .video-container {
            position: relative;
            width: 100%;
            margin-bottom: 15px;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.6);
            border: 1px solid #29292e;
        }
        #stream {
            width: 100%;
            display: block;
            cursor: crosshair;
        }
        .controls {
            width: 100%;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-bottom: 15px;
        }
        button {
            padding: 14px;
            font-size: 1rem;
            font-weight: bold;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: transform 0.1s, opacity 0.1s;
            color: white;
            text-transform: uppercase;
            box-shadow: 0 4px 6px rgba(0,0,0,0.15);
        }
        button:active {
            transform: scale(0.97);
            opacity: 0.9;
        }
        .btn-start {
            background: linear-gradient(135deg, #00b050, #008030);
            grid-column: span 2;
            padding: 18px;
            font-size: 1.2rem;
        }
        .btn-stop {
            background: linear-gradient(135deg, #cc0000, #990000);
            grid-column: span 2;
            padding: 18px;
            font-size: 1.2rem;
            margin-top: 5px;
        }
        .btn-hunt {
            background: linear-gradient(135deg, #7b1fa2, #4a148c);
            grid-column: span 2;
            padding: 14px;
            font-size: 1rem;
            margin-top: 5px;
        }
        .btn-servo { background: linear-gradient(135deg, #ff8c00, #e05c00); }
        .btn-reset { background: linear-gradient(135deg, #555, #333); }
        .btn-video {
            background: linear-gradient(135deg, #0088cc, #0066aa);
            grid-column: span 2;
            padding: 10px;
            font-size: 0.9rem;
            margin-top: 5px;
        }
        .status-card {
            width: 100%;
            background: #18181b;
            border: 1px solid #29292e;
            border-radius: 8px;
            padding: 12px;
            box-sizing: border-box;
            font-size: 0.9rem;
            line-height: 1.5;
            margin-top: 10px;
        }
        .status-item {
            display: flex;
            justify-content: space-between;
            margin-bottom: 6px;
            border-bottom: 1px solid #222;
            padding-bottom: 4px;
        }
        .status-item:last-child {
            margin-bottom: 0;
            border-bottom: none;
            padding-bottom: 0;
        }
        .val { font-weight: bold; color: #fff; }
    </style>
</head>
<body>
    <div class="container">
        <h1>NEYMARK Robot Web Control</h1>
        
        <div class="controls">
            <button class="btn-start" onclick="action('/start')">START (Автопилот)</button>
            <button class="btn-stop" onclick="action('/stop')">STOP (Останов)</button>
            <button class="btn-hunt" id="btn-hunt" onclick="toggleHunt()">HUNT MODE: ON</button>
            
            <button class="btn-servo" onclick="action('/test_servo')">Test Servo</button>
            <button class="btn-reset" onclick="action('/reset')">Reset Visited</button>
            
            <button class="btn-video" id="toggle-video-btn" onclick="toggleVideo()">Show Camera Stream</button>
        </div>

        <div class="video-container" id="video-box" style="display: none;">
            <img id="stream" src="" alt="Video Stream" onclick="sendClick(event)">
        </div>

        <div class="status-card">
            <div class="status-item"><span>Состояние:</span> <span class="val" id="status-state">-</span></div>
            <div class="status-item"><span>Команда:</span> <span class="val" id="status-cmd">-</span></div>
            <div class="status-item"><span>Цель:</span> <span class="val" id="status-target">-</span></div>
            <div class="status-item"><span>Посещено:</span> <span class="val" id="status-visited">-</span></div>
        </div>
    </div>
    <script>
        let videoActive = false;
        function toggleVideo() {
            const box = document.getElementById('video-box');
            const img = document.getElementById('stream');
            const btn = document.getElementById('toggle-video-btn');
            videoActive = !videoActive;
            if (videoActive) {
                box.style.display = 'block';
                img.src = '/video_feed';
                btn.innerText = 'Hide Camera Stream';
            } else {
                box.style.display = 'none';
                img.src = '';
                btn.innerText = 'Show Camera Stream';
            }
        }

        function toggleHunt() {
            fetch('/toggle_hunt', {method: 'POST'})
                .then(res => res.json())
                .then(data => {
                    const btn = document.getElementById('btn-hunt');
                    if (data.hunt_mode) {
                        btn.innerText = 'HUNT MODE: ON';
                        btn.style.background = 'linear-gradient(135deg, #7b1fa2, #4a148c)';
                    } else {
                        btn.innerText = 'HUNT MODE: OFF';
                        btn.style.background = 'linear-gradient(135deg, #444, #222)';
                    }
                });
        }

        function action(url) {
            fetch(url, {method: 'POST'})
                .then(res => res.json())
                .catch(err => console.error(err));
        }
        
        function sendClick(e) {
            const img = document.getElementById('stream');
            const rect = img.getBoundingClientRect();
            const x = (e.clientX - rect.left) / rect.width;
            const y = (e.clientY - rect.top) / rect.height;
            
            fetch('/click', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({x: x, y: y})
            })
            .then(res => res.json())
            .catch(err => console.error(err));
        }

        function updateStatus() {
            fetch('/status')
                .then(res => res.json())
                .then(data => {
                    document.getElementById('status-state').innerText = data.state;
                    document.getElementById('status-cmd').innerText = data.command;
                    document.getElementById('status-target').innerText = data.target_pixel ? `${Math.round(data.target_pixel[0])}, ${Math.round(data.target_pixel[1])}` : 'None';
                    document.getElementById('status-visited').innerText = data.visited_markers.join(', ') || 'None';
                })
                .catch(err => console.error(err));
        }

        setInterval(updateStatus, 300);
    </script>
</body>
</html>
"""

@app.route('/')
def index_route():
    return render_template_string(HTML_TEMPLATE)

def gen_frames():
    global latest_jpeg_frame
    while True:
        if latest_jpeg_frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + latest_jpeg_frame + b'\r\n')
        time.sleep(0.04) # ~25 FPS

@app.route('/video_feed')
def video_feed_route():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/click', methods=['POST'])
def click_route():
    global target_pixel, target_marker_id, servo_active, waiting_at_target
    data = request.get_json() or {}
    x_frac = data.get('x', 0.0)
    y_frac = data.get('y', 0.0)
    click_x = int(x_frac * frame_width)
    click_y = int(y_frac * frame_height)
    
    target_pixel = (click_x, click_y)
    target_marker_id = None
    servo_active = True
    waiting_at_target = False
    print(f"[WEB] Цель задана вручную с телефона/ПК: {target_pixel}")
    return jsonify({"status": "ok"})

@app.route('/start', methods=['POST'])
def web_start():
    global autopilot_enabled, servo_active
    autopilot_enabled = True
    servo_active = True
    print("[WEB] Автопилот ЗАПУЩЕН.")
    return jsonify({"status": "ok"})

@app.route('/toggle_hunt', methods=['POST'])
def web_toggle_hunt():
    global hunt_mode_enabled
    hunt_mode_enabled = not hunt_mode_enabled
    print(f"[WEB] Режим охоты: {'ВКЛЮЧЕН' if hunt_mode_enabled else 'ВЫКЛЮЧЕН'}")
    return jsonify({"status": "ok", "hunt_mode": hunt_mode_enabled})

@app.route('/test_servo', methods=['POST'])
def web_test_servo():
    global servo_active, servo_step, servo_pause_current
    servo_active = True
    servo_step = 0
    servo_pause_current = 0.5
    print("[WEB] Запуск теста сервопривода...")
    return jsonify({"status": "ok"})

@app.route('/stop', methods=['POST'])
def web_stop():
    global autopilot_enabled, target_pixel, target_marker_id, servo_active, waiting_at_target, avoid_state
    autopilot_enabled = False
    target_pixel = None
    target_marker_id = None
    servo_active = False
    waiting_at_target = False
    avoid_state = "NORMAL"
    print("[WEB] Экстренный останов.")
    return jsonify({"status": "ok"})

@app.route('/reset', methods=['POST'])
def web_reset():
    global visited_marker_ids
    visited_marker_ids.clear()
    print("[WEB] Список посещенных маркеров сброшен!")
    return jsonify({"status": "ok"})

@app.route('/status')
def web_status():
    global target_pixel, visited_marker_ids, state_str, current_command
    t_pixel = target_pixel
    v_ids = list(visited_marker_ids)
    return jsonify({
        "state": state_str,
        "command": current_command,
        "target_pixel": t_pixel,
        "visited_markers": v_ids
    })


def send(connection: socket.socket, command: str) -> None:
    try:
        connection.sendall((command + "\n").encode("ascii"))
    except (socket.timeout, OSError) as e:
        print(f"[ERROR] Ошибка отправки данных: {e}")


def send_servo(connection: socket.socket, angle: int) -> None:
    cmd = f"SERVO {angle}"
    print(f"[CMD SERVO] {cmd}")
    for _ in range(3):
        send(connection, cmd)
        time.sleep(0.015)


telemetry_buffer = ""

def drain_telemetry(connection: socket.socket) -> None:
    global left_line_sensor, right_line_sensor, telemetry_buffer
    connection.setblocking(False)
    try:
        data = connection.recv(4096).decode('ascii', errors='ignore')
        if data:
            telemetry_buffer += data
            if "\n" in telemetry_buffer:
                lines = telemetry_buffer.split("\n")
                telemetry_buffer = lines[-1]
                for line in reversed(lines[:-1]):
                    line = line.strip()
                    if line.startswith("TEL"):
                        parts = line.split()
                        if len(parts) >= 10:
                            try:
                                left_line_sensor = int(parts[8])
                                right_line_sensor = int(parts[9])
                            except ValueError:
                                pass
                        break
    except (BlockingIOError, socket.timeout, OSError):
        pass
    finally:
        connection.settimeout(0.05)


def mouse_callback(event: int, x: int, y: int, _flags: int, _data: object) -> None:
    global target_pixel, target_marker_id, servo_active, servo_step, servo_pause_current, waiting_at_target
    if event == cv2.EVENT_LBUTTONDOWN:
        if 20 <= x <= 160 and 100 <= y <= 135:
            servo_active = True
            servo_step = 0
            servo_pause_current = 0.5 # Медленная скорость при обычном ручном тесте
            print("[INFO] Клик по кнопке: Запуск теста сервопривода...")
            return

        target_pixel = (x, y)
        target_marker_id = None
        servo_active = True
        waiting_at_target = False
        print(f"[INFO] Цель задана вручную: {target_pixel}")
    elif event == cv2.EVENT_RBUTTONDOWN:
        target_pixel = None
        target_marker_id = None
        servo_active = False
        waiting_at_target = False
        print("[INFO] Цель сброшена.")


def robot_geometry(corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = corners.reshape(4, 2)
    center = points.mean(axis=0)
    front = 0.5 * (points[0] + points[1])
    
    if INVERT_ROBOT_HEADING:
        heading = center - front
    else:
        heading = front - center

    norm = np.linalg.norm(heading)
    if norm > 0:
        heading /= norm
    return center, heading


def get_marker_center(corners: np.ndarray) -> tuple[int, int]:
    points = corners.reshape(4, 2)
    center = points.mean(axis=0)
    return int(center[0]), int(center[1])


def signed_angle(heading: np.ndarray, target_vector: np.ndarray) -> float:
    hx, hy = float(heading[0]), -float(heading[1])
    tx, ty = float(target_vector[0]), -float(target_vector[1])

    cross = hx * ty - hy * tx
    dot = hx * tx + hy * ty
    angle = math.atan2(cross, dot)

    if abs(angle) > math.radians(165):
        return math.pi

    return angle


def select_next_target(
    robot_center: np.ndarray, 
    markers: dict[int, np.ndarray]
) -> tuple[tuple[int, int] | None, int | None]:
    best_marker_target = None
    best_marker_id = None
    min_marker_dist = float('inf')

    for m_id, m_corners in markers.items():
        if m_id == ROBOT_MARKER_ID or m_id in visited_marker_ids:
            continue
        
        mc_x, mc_y = get_marker_center(m_corners)
        dist = math.hypot(mc_x - robot_center[0], mc_y - robot_center[1])

        if dist < min_marker_dist:
            min_marker_dist = dist
            best_marker_target = (mc_x, mc_y)
            best_marker_id = m_id

    return best_marker_target, best_marker_id


def draw_map(robot_pos: tuple[int, int] | None) -> np.ndarray:
    grid_img = np.ones((MAP_SIZE[0], MAP_SIZE[1], 3), dtype=np.uint8) * 245
    
    for x in range(0, MAP_SIZE[1], GRID_STEP):
        cv2.line(grid_img, (x, 0), (x, MAP_SIZE[0]), (220, 220, 220), 1)
    for y in range(0, MAP_SIZE[0], GRID_STEP):
        cv2.line(grid_img, (0, y), (MAP_SIZE[1], y), (220, 220, 220), 1)
        
    if len(path_history) > 1:
        for i in range(1, len(path_history)):
            cv2.line(grid_img, path_history[i-1], path_history[i], (0, 100, 255), 2)
            
    if target_pixel is not None:
        cv2.circle(grid_img, target_pixel, 6, (0, 0, 255), -1)
        label = f"Target (ID:{target_marker_id})" if target_marker_id is not None else "Target"
        cv2.putText(grid_img, label, (target_pixel[0] + 10, target_pixel[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
    if robot_pos is not None:
        cv2.circle(grid_img, robot_pos, 8, (255, 0, 0), -1)
        cv2.putText(grid_img, "Robot", (robot_pos[0] + 10, robot_pos[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
    cv2.putText(grid_img, f"Path points: {len(path_history)} | Visited: {len(visited_marker_ids)}", 
                (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 50, 50), 1)
    
    return grid_img


def main() -> None:
    global target_pixel, target_marker_id, visited_marker_ids, path_history
    global last_robot_center, last_robot_heading, last_robot_see_time
    global servo_active, servo_step, servo_step_time, servo_pause_current
    global frame_width, frame_height, state_str, current_command, latest_jpeg_frame
    global autopilot_enabled, waiting_at_target, target_arrival_time, hunt_mode_enabled
    global avoid_state, avoid_end_time, avoid_turn_dir, left_line_sensor, right_line_sensor

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(
        dictionary,
        cv2.aruco.DetectorParameters(),
    )

    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        raise RuntimeError("Ошибка: Камера не открылась.")

    ok, frame_init = camera.read()
    if ok:
        frame_height, frame_width = frame_init.shape[:2]

    print(f"[INFO] Подключение к роботу по адресу {ROBOT_ADDRESS}...")
    connection = socket.create_connection(ROBOT_ADDRESS, timeout=3.0)
    try:
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except OSError:
        pass
    connection.settimeout(0.05)

    window_cam = "Camera View"
    window_map = "Coordinate Grid & Path"
    cv2.namedWindow(window_cam)
    cv2.namedWindow(window_map)
    
    cv2.setMouseCallback(window_cam, mouse_callback)
    previous_send_time = 0.0
    last_sent_command = ""

    send_servo(connection, SERVO_CENTER_DEG)

    ips = []
    try:
        import subprocess
        import re
        output_net = subprocess.check_output(['ifconfig'], text=True)
        ips = re.findall(r'inet\s+(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', output_net)
        ips = [ip for ip in ips if not ip.startswith('127.')]
    except Exception:
        pass

    if not ips:
        ips = ["127.0.0.1"]

    def run_flask():
        app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)

    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    print("[INFO] Веб-интерфейс запущен! Откройте в браузере телефона/ПК:")
    for ip in ips:
        print(f"  -> http://{ip}:5001")
    print("[READY] Готово. 'R' - сброс посещенных маркеров, 'SPACE' - стоп.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("[ERROR] Не удалось получить кадр с камеры.")
                break

            now_time = time.monotonic()
            drain_telemetry(connection)

            corners, ids, _rejected = detector.detectMarkers(frame)
            markers: dict[int, np.ndarray] = {}
            robot_pos_for_map = None

            if ids is not None:
                markers = {
                    int(marker_id): marker_corners
                    for marker_corners, marker_id in zip(corners, ids.flatten())
                }
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            if ROBOT_MARKER_ID in markers:
                last_robot_center, last_robot_heading = robot_geometry(markers[ROBOT_MARKER_ID])
                last_robot_see_time = now_time

            robot_is_valid = (now_time - last_robot_see_time) <= MARKER_TIMEOUT_SEC
            robot_center = last_robot_center if robot_is_valid else None
            robot_heading = last_robot_heading if robot_is_valid else None

            detected_enemy_id = None
            detected_enemy_pixel = None
            is_attacking = False

            if hunt_mode_enabled and avoid_state == "NORMAL":
                for e_id in ENEMY_MARKER_IDS:
                    if e_id in markers:
                        detected_enemy_id = e_id
                        detected_enemy_pixel = get_marker_center(markers[e_id])
                        is_attacking = True
                        break

            if is_attacking and avoid_state == "NORMAL":
                target_pixel = detected_enemy_pixel
                target_marker_id = detected_enemy_id
                waiting_at_target = False
            else:
                if target_marker_id in ENEMY_MARKER_IDS and avoid_state == "NORMAL":
                    target_pixel = None
                    target_marker_id = None

            command = "STOP"
            state = "SEARCHING TARGETS"

            if avoid_state == "NORMAL":
                if left_line_sensor > LINE_THRESHOLD or right_line_sensor > LINE_THRESHOLD:
                    avoid_state = "BACKING_UP"
                    avoid_end_time = now_time + 0.65
                    avoid_turn_dir = -1 if left_line_sensor > LINE_THRESHOLD else 1
                    waiting_at_target = False
                    print(f"[LINE AVOID] Линия! L={left_line_sensor}, R={right_line_sensor}. Откат назад...")

            if avoid_state == "BACKING_UP":
                if now_time < avoid_end_time:
                    command = "VEL -200 0"
                    state = "LINE AVOID: BACKUP"
                else:
                    avoid_state = "TURNING_AWAY"
                    avoid_end_time = now_time + 0.5

            if avoid_state == "TURNING_AWAY":
                if now_time < avoid_end_time:
                    angular = 3500 if avoid_turn_dir == 1 else -3500
                    command = f"VEL 0 {angular}"
                    state = "LINE AVOID: TURN"
                else:
                    avoid_state = "NORMAL"
                    command = "STOP"
                    state = "SEARCHING TARGETS"
                    print("[LINE AVOID] Возврат к нормальному движению.")

            if waiting_at_target and avoid_state == "NORMAL":
                if now_time - target_arrival_time >= 0.8:
                    target_pixel = None
                    target_marker_id = None
                    waiting_at_target = False
                    print("[INFO] Сканирование завершено, переходим к следующей цели.")
                else:
                    command = "STOP"
                    state = f"SCANNING #{target_marker_id}"

            if robot_center is not None and robot_heading is not None:
                robot_pixel = tuple(np.rint(robot_center).astype(int))
                robot_pos_for_map = robot_pixel

                arrow_color = (0, 255, 0) if ROBOT_MARKER_ID in markers else (0, 255, 255)
                nose_pt = (
                    int(robot_pixel[0] + robot_heading[0] * 35),
                    int(robot_pixel[1] + robot_heading[1] * 35)
                )
                cv2.arrowedLine(frame, robot_pixel, nose_pt, arrow_color, 2)

            for m_id, m_corners in markers.items():
                if m_id == ROBOT_MARKER_ID:
                    continue
                mc = get_marker_center(m_corners)
                if m_id in ENEMY_MARKER_IDS:
                    cv2.circle(frame, mc, 15, (0, 0, 255), 2)
                    cv2.putText(frame, f"ENEMY #{m_id}", (mc[0] - 40, mc[1] - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                else:
                    color = (128, 128, 128) if m_id in visited_marker_ids else (255, 255, 0)
                    status_str = "VISITED" if m_id in visited_marker_ids else f"MARKER #{m_id}"
                    cv2.putText(frame, status_str, (mc[0] - 30, mc[1] - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if autopilot_enabled and robot_center is not None and target_pixel is None and not waiting_at_target and avoid_state == "NORMAL":
                new_target, new_target_id = select_next_target(robot_center, markers)
                if new_target is not None:
                    target_pixel = new_target
                    target_marker_id = new_target_id
                    servo_active = True
                    print(f"[INFO] Выбран новый маркер-цель #{target_marker_id}: {target_pixel}")

            if target_marker_id is not None and target_marker_id in markers and not waiting_at_target and avoid_state == "NORMAL":
                target_pixel = get_marker_center(markers[target_marker_id])

            if target_pixel is not None and robot_center is not None and avoid_state == "NORMAL":
                target_pos = np.array(target_pixel, dtype=np.float32)
                dist_to_target = float(np.linalg.norm(target_pos - robot_center))
                if dist_to_target <= TARGET_TOLERANCE_PX:
                    servo_pause_current = 0.5 / 3.0
                else:
                    servo_pause_current = 0.5
            else:
                servo_pause_current = 0.5

            if servo_active:
                if servo_step == 0:
                    send_servo(connection, SERVO_MAX_DEG)
                    servo_step = 1
                    servo_step_time = now_time
                elif now_time - servo_step_time >= servo_pause_current:
                    if servo_step == 1:
                        send_servo(connection, SERVO_CENTER_DEG)
                        servo_step = 2
                        servo_step_time = now_time
                    elif servo_step == 2:
                        send_servo(connection, SERVO_MIN_DEG)
                        servo_step = 3
                        servo_step_time = now_time
                    elif servo_step == 3:
                        send_servo(connection, SERVO_CENTER_DEG)
                        servo_step = 4
                        servo_step_time = now_time
                    elif servo_step == 4:
                        servo_step = 0

            if autopilot_enabled and target_pixel is not None and not waiting_at_target and avoid_state == "NORMAL":
                cv2.circle(frame, target_pixel, TARGET_TOLERANCE_PX, (0, 170, 255), 2)

                if not robot_is_valid:
                    state = f"LOST ROBOT MARKER {ROBOT_MARKER_ID}"
                    command = "STOP"
                else:
                    if not path_history or math.hypot(robot_pixel[0]-path_history[-1][0], robot_pixel[1]-path_history[-1][1]) > 5:
                        path_history.append(robot_pixel)

                    target = np.array(target_pixel, dtype=np.float32)
                    target_vector = target - robot_center
                    distance = float(np.linalg.norm(target_vector))
                    angle_error = signed_angle(robot_heading, target_vector)

                    line_color = (0, 0, 255) if is_attacking else (255, 80, 0)
                    cv2.arrowedLine(frame, robot_pixel, target_pixel, line_color, 3)

                    if not is_attacking and distance <= TARGET_TOLERANCE_PX:
                        command = "STOP"
                        state = "TARGET REACHED"
                        send(connection, "STOP")

                        if target_marker_id is not None:
                            print(f"[INFO] Маркер #{target_marker_id} достигнут!")
                            visited_marker_ids.add(target_marker_id)
                        
                        waiting_at_target = True
                        target_arrival_time = now_time

                        servo_active = True
                        servo_step = 0
                        servo_pause_current = 0.5 / 3.0

                    elif is_attacking and distance <= TARGET_TOLERANCE_PX:
                        command = f"VEL {ATTACK_SPEED_MM_S} 0"
                        state = f"RAMMING ENEMY #{target_marker_id}!"
                        servo_pause_current = 0.5 / 3.0

                    elif abs(angle_error) > math.radians(45):
                        angular = ANGULAR_SPEED_MRAD_S if angle_error > 0 else -ANGULAR_SPEED_MRAD_S
                        command = f"VEL 0 {angular}"
                        state = "TURN" if not is_attacking else "ATTACK TURN"

                    else:
                        angular_speed_mrad = int(STEERING_KP * angle_error * 1000)
                        angular_speed_mrad = max(min(angular_speed_mrad, ANGULAR_SPEED_MRAD_S), -ANGULAR_SPEED_MRAD_S)
                        
                        speed = LINEAR_SPEED_MM_S if not is_attacking else ATTACK_SPEED_MM_S
                        command = f"VEL {speed} {angular_speed_mrad}"
                        state = "DRIVE (ARC)" if not is_attacking else "ATTACK DRIVE"

                    cv2.putText(
                        frame,
                        f"distance={distance:.0f}px  error={math.degrees(angle_error):.1f}deg",
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                    )

            if not autopilot_enabled:
                command = "STOP"
                state = "STOPPED"

            cv2.putText(
                frame,
                state,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0) if command != "STOP" else (0, 100, 255),
                2,
            )
            cv2.rectangle(frame, (20, 100), (160, 135), (0, 120, 240), -1)
            cv2.rectangle(frame, (20, 100), (160, 135), (255, 255, 255), 1)
            cv2.putText(
                frame,
                "TEST SERVO",
                (35, 123),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                "LMB: target | RMB/SPACE: stop | R: reset visited | ESC: exit",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )

            if now_time - previous_send_time >= SEND_PERIOD_SECONDS:
                send(connection, command)
                if command != last_sent_command:
                    print(f"[CMD] Sent: '{command}' | State: {state}")
                    last_sent_command = command
                previous_send_time = now_time

            map_frame = draw_map(robot_pos_for_map)
            
            state_str = state
            current_command = command
            try:
                _, jpeg_bytes = cv2.imencode('.jpg', frame)
                latest_jpeg_frame = jpeg_bytes.tobytes()
            except Exception:
                pass

            cv2.imshow(window_cam, frame)
            cv2.imshow(window_map, map_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == ord("r") or key == ord("R"):
                visited_marker_ids.clear()
                print("[INFO] Список посещенных маркеров сброшен!")
            elif key == ord(" "):
                target_pixel = None
                target_marker_id = None
                path_history.clear()
                servo_active = False
                autopilot_enabled = False
                waiting_at_target = False
                avoid_state = "NORMAL"
                send(connection, "STOP")
                print("[INFO] Экстренный останов.")

    finally:
        print("[INFO] Завершение работы...")
        send(connection, "STOP")
        connection.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()