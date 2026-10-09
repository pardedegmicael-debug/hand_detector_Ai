"""
Hand Motion Tracker - pelacakan tangan dan gerakan tangan secara detail.

Fitur utama:
- Mendeteksi sampai 2 tangan dengan 21 landmark per tangan.
- Menampilkan skeleton tangan, koordinat landmark, bounding box, dan trail gerak.
- Menghitung sudut sendi jari, status setiap jari, gesture dasar, serta pinch distance.
- Menghitung perpindahan, kecepatan pixel/detik, kecepatan ternormalisasi, dan arah gerak.
- Menyimpan data mentah dan hasil analisis setiap frame ke file CSV.
- Pose pointing yang ditahan selama 1,5 detik keluar dari aplikasi.
- Pose rock memutar audio; pose jari tengah selama 1,5 detik memulai hitung mundur lokal.
- Shutdown Windows baru dikirim setelah hitung mundur 15 detik selesai.
- Gesture keluar/shutdown hanya aktif untuk webcam, bukan rekaman video.
- Audio bersifat opsional; audio hilang/rusak tidak menghentikan pelacakan.
- Frame tracker dapat dikirim ke kamera virtual dengan opsi --virtual-camera.

Kontrol saat program berjalan:
  q / ESC : keluar; batalkan hitung mundur lokal sebelum perintah shutdown dikirim
  x       : batalkan hitung mundur tanpa keluar dari tracker
  pointing: tahan 1,5 detik untuk keluar dari aplikasi
  c       : hapus trail dan riwayat kecepatan
  s       : simpan screenshot
  d       : tampilkan/sembunyikan nomor landmark

Instalasi dan lingkungan yang valid:
  Python 3.11/3.12: python -m pip install "mediapipe==0.10.21" "opencv-contrib-python<4.12" "numpy<2" pygame
  Jika Anda memakai Python 3.13+, install modern MediaPipe bukan legacy API, atau gunakan venv 3.11/3.12 yang sudah
  disiapkan di folder proyek (.venv311). Skrip ini memakai API legacy mp.solutions.hands; jangan upgrade MediaPipe sembarangan.
  Audio opsional: rock.mp3 dan blur.mp3 di folder skrip atau lewat --audio/--blur-audio.
  OBS opsional: python -m pip install pyvirtualcam

Contoh:
  .venv311/Scripts/python.exe hand_motion_tracker.py
  .venv311/Scripts/python.exe hand_motion_tracker.py --camera 0 --width 1280 --height 720
  .venv311/Scripts/python.exe hand_motion_tracker.py --video rekaman.mp4 --no-display
  .venv311/Scripts/python.exe hand_motion_tracker.py --no-csv
  .venv311/Scripts/python.exe hand_motion_tracker.py --virtual-camera
  .venv311/Scripts/python.exe hand_motion_tracker.py --virtual-camera --virtual-camera-backend unitycapture
  .venv311/Scripts/python.exe hand_motion_tracker.py --disable-shutdown --no-audio
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
import time
from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


# Indeks landmark MediaPipe Hands.
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
FINGER_POINTS = {
    "thumb": (THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP),
    "index": (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    "pinky": (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
}
TIP_INDEX = {"thumb": THUMB_TIP, "index": INDEX_TIP, "middle": MIDDLE_TIP,
             "ring": RING_TIP, "pinky": PINKY_TIP}

LANDMARK_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


Point = Tuple[float, float]

SHUTDOWN_COUNTDOWN_SECONDS = 15
SHUTDOWN_POSE_HOLD_SECONDS = 1.5
POINTING_EXIT_HOLD_SECONDS = 1.5


@dataclass
class HandState:
    """Riwayat analisis satu tangan agar gerak bisa dihitung antar-frame."""

    previous_center: Optional[Point] = None
    previous_time: Optional[float] = None
    smoothed_velocity: Point = (0.0, 0.0)
    trail: deque = field(default_factory=lambda: deque(maxlen=80))
    speed_history: deque = field(default_factory=lambda: deque(maxlen=12))
    handedness: str = "Unknown"


@dataclass
class HandTracker:
    """Pasangkan deteksi antar-frame; label Left/Right bukan ID unik."""

    states: Dict[int, HandState] = field(default_factory=dict)
    next_id: int = 1

    def associate(self, detections: List[Tuple[str, Point]], frame_diagonal: float) -> List[int]:
        max_jump = max(frame_diagonal * 0.25, 1.0)
        best_assignment = (None,) * len(detections)
        best_cost = float("inf")
        # Maksimal dua tangan: cari pasangan global agar urutan deteksi tidak berpengaruh.
        for assignment in product((None, *self.states), repeat=len(detections)):
            used = [track_id for track_id in assignment if track_id is not None]
            if len(used) != len(set(used)):
                continue
            cost = 0.0
            for (label, center), track_id in zip(detections, assignment):
                if track_id is None:
                    cost += max_jump
                    continue
                state = self.states[track_id]
                jump = float("inf") if state.previous_center is None else distance(center, state.previous_center)
                if jump > max_jump:
                    cost = float("inf")
                    break
                cost += jump + (0.1 * max_jump if label != state.handedness else 0.0)
            if cost < best_cost:
                best_assignment, best_cost = assignment, cost
        visible_states = {}
        track_ids = []
        for (label, _), track_id in zip(detections, best_assignment):
            if track_id is None:
                track_id = self.next_id
                self.next_id += 1
                state = HandState()
            else:
                state = self.states[track_id]
            state.handedness = label
            visible_states[track_id] = state
            track_ids.append(track_id)
        # Tangan hilang memutus trail, kecepatan, dan timer gesture.
        self.states = visible_states
        return track_ids


@dataclass
class PoseHold:
    since: Dict[int, float] = field(default_factory=dict)

    def elapsed(self, hand_ids: Iterable[int], now: float) -> Optional[float]:
        self.since = {key: self.since.get(key, now) for key in hand_ids}
        if not self.since:
            return None
        return max(0.0, now - min(self.since.values()))


@dataclass
class VideoClock:
    """Waktu rekaman, terpisah dari waktu pemrosesan komputer."""

    fps: float
    previous: Optional[float] = None
    warned: bool = False

    def timestamp(self, position_ms: float) -> float:
        seconds = position_ms / 1000.0
        if math.isfinite(seconds) and seconds >= 0 and (self.previous is None or seconds > self.previous):
            self.previous = seconds
            return seconds
        if not math.isfinite(self.fps) or self.fps <= 0:
            if not self.warned:
                print("PERINGATAN: timestamp/FPS video tidak valid; waktu diperkirakan dengan 30 FPS.")
                self.warned = True
            interval = 1.0 / 30.0
        else:
            interval = 1.0 / self.fps
        self.previous = 0.0 if self.previous is None else self.previous + interval
        return self.previous


@dataclass
class ShutdownCountdown:
    """Hitung mundur lokal; tidak mengirim perintah Windows sebelum waktunya habis."""

    hold: PoseHold = field(default_factory=PoseHold)
    deadline: Optional[float] = None
    command_sent: bool = False
    require_pose_release: bool = False

    def observe(self, hand_ids: Iterable[int], now: float) -> bool:
        if self.command_sent or self.deadline is not None:
            return False
        hand_ids = tuple(hand_ids)
        if not hand_ids:
            self.hold.since.clear()
            self.require_pose_release = False
            return False
        if self.require_pose_release:
            return False
        elapsed = self.hold.elapsed(hand_ids, now)
        if elapsed is None or elapsed < SHUTDOWN_POSE_HOLD_SECONDS:
            return False
        self.deadline = now + SHUTDOWN_COUNTDOWN_SECONDS
        self.hold.since.clear()
        return True

    def remaining(self, now: float) -> Optional[int]:
        if self.deadline is None:
            return None
        return max(0, math.ceil(self.deadline - now))

    def cancel(self) -> None:
        self.deadline = None
        self.hold.since.clear()
        self.require_pose_release = True

    def submit_if_due(self, now: float) -> bool:
        if self.command_sent or self.deadline is None or now < self.deadline:
            return False
        # Bersihkan hitungan dahulu; jika perintah gagal, jangan ulangi tiap frame.
        self.cancel()
        try:
            run_shutdown_command("/s", "/t", "0", "/f")
        except RuntimeError:
            self.command_sent = False
            raise
        self.command_sent = True
        return True


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pelacak gerakan tangan detail berbasis webcam")
    parser.add_argument("--camera", type=int, default=0, help="ID kamera, default: 0")
    parser.add_argument("--video", type=Path, help="File video yang akan dianalisis; jika diisi, kamera diabaikan")
    parser.add_argument("--width", type=int, default=1280, help="Lebar kamera")
    parser.add_argument("--height", type=int, default=720, help="Tinggi kamera")
    parser.add_argument("--max-hands", type=int, default=2, choices=(1, 2), help="Jumlah tangan maksimal")
    parser.add_argument("--detection-confidence", type=float, default=0.70)
    parser.add_argument("--tracking-confidence", type=float, default=0.70)
    parser.add_argument("--csv", default="hand_motion_log.csv", help="Nama file CSV keluaran")
    parser.add_argument("--audio", type=Path, help="Jalur file audio MP3 untuk pose rock; bila tidak diisi, akan otomatis dicari di folder proyek/venv")
    parser.add_argument("--blur-audio", type=Path, help="Jalur file audio MP3 untuk pose peace; bila tidak diisi, akan otomatis dicari di folder proyek/venv")
    parser.add_argument("--no-csv", action="store_true", help="Jangan menyimpan data CSV")
    parser.add_argument("--no-audio", action="store_true", help="Matikan audio; pygame tidak diperlukan")
    parser.add_argument("--disable-shutdown", action="store_true", help="Matikan gesture shutdown untuk sesi ini")
    parser.add_argument("--no-mirror", action="store_true", help="Jangan membalik gambar secara horizontal")
    parser.add_argument("--no-display", action="store_true", help="Proses tanpa membuka jendela; cocok untuk video")
    parser.add_argument(
        "--virtual-camera",
        action="store_true",
        help="Kirim video hasil tracker ke kamera virtual (hanya untuk webcam langsung)",
    )
    parser.add_argument(
        "--virtual-camera-backend",
        choices=("obs", "unitycapture"),
        default="obs",
        help="Backend kamera virtual; gunakan unitycapture sebagai sumber video di OBS (default: obs)",
    )
    args = parser.parse_args(argv)
    if args.camera < 0:
        parser.error("--camera harus bernilai 0 atau lebih")
    if args.width <= 0 or args.height <= 0:
        parser.error("--width dan --height harus bernilai positif")
    if not 0.0 <= args.detection_confidence <= 1.0:
        parser.error("--detection-confidence harus berada di antara 0 dan 1")
    if not 0.0 <= args.tracking_confidence <= 1.0:
        parser.error("--tracking-confidence harus berada di antara 0 dan 1")
    if args.virtual_camera and args.video is not None:
        parser.error("--virtual-camera hanya dapat digunakan dengan webcam langsung, bukan --video")
    if args.video is not None:
        args.video = Path(args.video).expanduser()
    return args


def find_audio_file(filename: str, start_dir: Optional[Path] = None) -> Optional[Path]:
    base_dir = start_dir or Path(__file__).resolve().parent
    candidates = [base_dir / filename]
    for root in (base_dir, *[base_dir / name for name in (".venv", ".venv311")]):
        candidates.extend((root / "Include" / filename, root / "Lib" / "site-packages" / filename))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    # Baru telusuri satu kali jika semua lokasi umum tidak berisi audio.
    try:
        for candidate in base_dir.rglob(filename):
            if candidate.is_file():
                return candidate.resolve()
    except OSError as exc:
        print(f"PERINGATAN: pencarian audio {filename} gagal: {exc}")
    return None


def find_rock_audio_file(start_dir: Optional[Path] = None) -> Optional[Path]:
    return find_audio_file("rock.mp3", start_dir)


def find_blur_audio_file(start_dir: Optional[Path] = None) -> Optional[Path]:
    return find_audio_file("blur.mp3", start_dir)


class GestureAudio:
    """Satu kanal musik; peace mendapat prioritas dan diputar sekali per pose."""

    def __init__(self):
        self.pygame = None
        self.tracks: Dict[str, Path] = {}
        self.active: Optional[str] = None

    def start(self, args: argparse.Namespace) -> None:
        if args.no_audio:
            return
        for name, explicit, finder in (("rock", args.audio, find_rock_audio_file),
                                       ("peace", args.blur_audio, find_blur_audio_file)):
            path = explicit.expanduser().resolve() if explicit is not None else finder()
            if path is not None and path.is_file():
                self.tracks[name] = path
            else:
                print(f"PERINGATAN: audio {name} tidak ditemukan; tracker tetap berjalan.")
        if not self.tracks:
            return
        try:
            import pygame
        except ImportError:
            print("PERINGATAN: pygame tidak tersedia; audio dimatikan. Pasang dengan python -m pip install pygame.")
            return
        self.pygame = pygame
        try:
            pygame.mixer.init()
        except (pygame.error, OSError) as exc:
            print(f"PERINGATAN: perangkat audio gagal disiapkan: {exc}. Tracker tetap berjalan.")
            self.close()

    def update(self, peace: bool, rock: bool) -> None:
        if self.pygame is None:
            return
        desired = "peace" if peace and "peace" in self.tracks else (
            "rock" if rock and "rock" in self.tracks else None)
        if desired == self.active:
            return
        try:
            self.pygame.mixer.music.stop()
            self.active = None
            if desired is not None:
                self.pygame.mixer.music.load(str(self.tracks[desired]))
                self.pygame.mixer.music.play(loops=0 if desired == "peace" else -1)
                self.active = desired
        except (self.pygame.error, OSError) as exc:
            print(f"PERINGATAN: audio {desired or 'stop'} gagal: {exc}. Tracker tetap berjalan.")
            if desired is not None:
                self.tracks.pop(desired, None)
            else:
                self.close()

    def close(self) -> None:
        if self.pygame is None:
            return
        try:
            if self.pygame.mixer.get_init() is not None:
                try:
                    self.pygame.mixer.music.stop()
                finally:
                    self.pygame.mixer.quit()
        except (self.pygame.error, OSError) as exc:
            print(f"PERINGATAN: penutupan audio gagal: {exc}")
        finally:
            self.pygame = None
            self.active = None


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def angle_degrees(a: Point, b: Point, c: Point) -> float:
    """Sudut ABC dalam derajat."""
    ba = np.array([a[0] - b[0], a[1] - b[1]], dtype=float)
    bc = np.array([c[0] - b[0], c[1] - b[1]], dtype=float)
    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denominator < 1e-8:
        return 0.0
    cosine = float(np.dot(ba, bc) / denominator)
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def normalized_landmarks(hand_landmarks) -> List[Tuple[float, float, float]]:
    return [(float(point.x), float(point.y), float(point.z)) for point in hand_landmarks.landmark]


def pixel_landmarks(points: Iterable[Tuple[float, float, float]], width: int, height: int) -> List[Point]:
    # Jangan bulatkan/clip sebelum analisis: geometri rusak saat tangan di tepi frame.
    return [(x * width, y * height) for x, y, _ in points]


def palm_center(points: List[Point]) -> Point:
    # Menggunakan wrist dan empat MCP supaya pusat lebih stabil daripada satu titik saja.
    selected = [points[i] for i in (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)]
    return (sum(p[0] for p in selected) / len(selected), sum(p[1] for p in selected) / len(selected))


def finger_angles(points: List[Point]) -> Dict[str, float]:
    angles: Dict[str, float] = {}
    # Sudut PIP untuk empat jari utama. Sudut besar berarti lebih lurus.
    for name in ("index", "middle", "ring", "pinky"):
        mcp, pip, dip, tip = FINGER_POINTS[name]
        angles[f"{name}_pip_angle"] = angle_degrees(points[mcp], points[pip], points[dip])
        angles[f"{name}_dip_angle"] = angle_degrees(points[pip], points[dip], points[tip])
    angles["thumb_ip_angle"] = angle_degrees(points[THUMB_MCP], points[THUMB_IP], points[THUMB_TIP])
    return angles


def finger_states(points: List[Point], handedness: str) -> Dict[str, bool]:
    """Deteksi 2D berbasis sudut dan proporsi telapak, tidak bergantung resolusi."""
    center = palm_center(points)
    palm_size = max(distance(points[WRIST], points[MIDDLE_MCP]), 1.0)
    wrist_point = points[WRIST]
    result: Dict[str, bool] = {}

    for name in ("index", "middle", "ring", "pinky"):
        mcp, pip, dip, tip = FINGER_POINTS[name]
        angle = angle_degrees(points[mcp], points[pip], points[dip])
        dip_angle = angle_degrees(points[pip], points[dip], points[tip])
        tip_distance = distance(points[tip], center)
        base_distance = distance(points[mcp], center)
        tip_to_wrist = distance(points[tip], wrist_point)
        mcp_to_wrist = distance(points[mcp], wrist_point)

        extended_by_distance = tip_distance > max(base_distance + palm_size * 0.04, palm_size * 0.50)
        extended_by_length = tip_to_wrist > max(mcp_to_wrist * 1.10, palm_size * 0.72)
        not_too_folded = angle > 130.0 and dip_angle > 110.0
        result[name] = (extended_by_distance or extended_by_length) and not_too_folded

    thumb_tip_to_center = distance(points[THUMB_TIP], center)
    thumb_mcp_to_center = distance(points[THUMB_MCP], center)
    thumb_extension = distance(points[THUMB_TIP], points[INDEX_MCP]) / palm_size
    thumb_folded = distance(points[THUMB_TIP], points[THUMB_MCP]) < 0.68 * palm_size
    result["thumb"] = (thumb_tip_to_center > thumb_mcp_to_center + palm_size * 0.02) and (thumb_extension > 0.38) and not thumb_folded
    return result


def classify_gesture(points: List[Point], states: Dict[str, bool]) -> str:
    open_names = [name for name in FINGER_NAMES if states[name]]
    count = len(open_names)
    palm_size = max(distance(points[WRIST], points[MIDDLE_MCP]), 1.0)
    pinch_ratio = distance(points[THUMB_TIP], points[INDEX_TIP]) / palm_size
    # Fist hanya bila semua jari tertutup, bukan bila ada 1-2 jari yang agak menekuk.
    if count == 0:
        return "fist"

    if states["middle"] and not states["index"] and not states["ring"] and not states["pinky"]:
        return "fuck"
    if pinch_ratio < 0.28:
        return "pinch / OK"

    if states["thumb"] and not any(states[n] for n in ("index", "middle", "ring", "pinky")):
        return "thumbs up"
    if states["index"] and not states["middle"] and not states["ring"] and not states["pinky"]:
        return "pointing"
    if states["index"] and states["middle"] and not states["ring"] and not states["pinky"]:
        return "peace"
    if states["index"] and states["pinky"] and not states["middle"] and not states["ring"]:
        return "rock"
    if count == 5:
        return "open palm"
    return f"{count} fingers open"


def is_shutdown_pose(points: List[Point], states: Dict[str, bool]) -> bool:
    """Gerbang lebih ketat untuk aksi shutdown daripada label gesture di layar."""
    if not states["middle"] or any(states[name] for name in ("index", "ring", "pinky")):
        return False
    angles = finger_angles(points)
    palm_size = max(distance(points[WRIST], points[MIDDLE_MCP]), 1.0)
    return (
        angles["middle_pip_angle"] >= 155.0
        and angles["middle_dip_angle"] >= 150.0
        and distance(points[MIDDLE_TIP], points[WRIST])
        > distance(points[MIDDLE_PIP], points[WRIST]) + 0.12 * palm_size
        and all(angles[f"{name}_pip_angle"] < 130.0 for name in ("index", "ring", "pinky"))
    )


def action_permissions(args: argparse.Namespace) -> Tuple[bool, bool]:
    live_gestures = args.video is None and not args.virtual_camera
    return live_gestures, live_gestures and os.name == "nt" and not args.disable_shutdown


def run_shutdown_command(*arguments: str) -> None:
    if os.name != "nt":
        raise RuntimeError("Fitur shutdown hanya tersedia di Windows.")
    shutdown_exe = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "shutdown.exe"
    try:
        subprocess.run(
            [str(shutdown_exe), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        detail = getattr(exc, "stderr", None) or getattr(exc, "stdout", None) or str(exc)
        raise RuntimeError(f"Perintah shutdown Windows gagal: {detail}") from exc


def direction_from_velocity(vx: float, vy: float, threshold: float = 35.0) -> str:
    speed = math.hypot(vx, vy)
    if speed < threshold:
        return "diam"
    horizontal = "kanan" if vx > 0 else "kiri"
    vertical = "bawah" if vy > 0 else "atas"
    if abs(vx) < 0.45 * abs(vy):
        return vertical
    if abs(vy) < 0.45 * abs(vx):
        return horizontal
    return f"{vertical}-{horizontal}"


def update_motion(state: HandState, center: Point, now: float) -> Tuple[float, float, float, str]:
    if state.previous_center is None or state.previous_time is None:
        state.previous_center = center
        state.previous_time = now
        state.trail.append(center)
        return 0.0, 0.0, 0.0, "diam"

    dt = now - state.previous_time
    if dt <= 0:
        # Timestamp duplikat tidak boleh menghasilkan kecepatan sangat besar.
        reset_motion(state)
        return update_motion(state, center, now)
    raw_vx = (center[0] - state.previous_center[0]) / dt
    raw_vy = (center[1] - state.previous_center[1]) / dt

    # EMA mengurangi lonjakan akibat noise kamera tanpa menghilangkan gerakan cepat.
    alpha = 0.35
    vx = alpha * raw_vx + (1.0 - alpha) * state.smoothed_velocity[0]
    vy = alpha * raw_vy + (1.0 - alpha) * state.smoothed_velocity[1]
    state.smoothed_velocity = (vx, vy)
    speed = math.hypot(vx, vy)
    state.speed_history.append(speed)
    state.previous_center = center
    state.previous_time = now
    state.trail.append(center)
    return vx, vy, speed, direction_from_velocity(vx, vy)


def reset_motion(state: HandState) -> None:
    state.previous_center = None
    state.previous_time = None
    state.smoothed_velocity = (0.0, 0.0)
    state.trail.clear()
    state.speed_history.clear()


def correct_handedness(label: str, mirrored_input: bool) -> str:
    if mirrored_input:
        return label
    return {"Left": "Right", "Right": "Left"}.get(label, label)


def apply_peace_blur(frame: np.ndarray, active: bool = False) -> np.ndarray:
    if not active:
        return frame
    blurred = cv2.GaussianBlur(frame, (0, 0), 15)
    return cv2.addWeighted(blurred, 0.75, frame, 0.25, 0)


def draw_text(frame, text: str, origin: Tuple[int, int], color=(255, 255, 255), scale=0.52, thickness=1):
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_trail(frame, trail: deque, color: Tuple[int, int, int]):
    trail_list = list(trail)
    for i in range(1, len(trail_list)):
        fade = i / max(len(trail_list), 1)
        thickness = max(1, int(1 + 4 * fade))
        p1 = tuple(map(int, trail_list[i - 1]))
        p2 = tuple(map(int, trail_list[i]))
        cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)


def draw_hand(frame, points: List[Point], label: str, color: Tuple[int, int, int], state: HandState,
              show_details: bool, velocity: Tuple[float, float, float, str]):
    height, width = frame.shape[:2]
    center = palm_center(points)
    cx, cy = map(int, center)
    vx, vy, speed, direction = velocity
    xs = [int(p[0]) for p in points]
    ys = [int(p[1]) for p in points]
    x1, x2 = max(min(xs) - 15, 0), min(max(xs) + 15, width - 1)
    y1, y2 = max(min(ys) - 15, 0), min(max(ys) + 15, height - 1)

    draw_trail(frame, state.trail, color)
    for a, b in LANDMARK_CONNECTIONS:
        cv2.line(frame, tuple(map(int, points[a])), tuple(map(int, points[b])), color, 2, cv2.LINE_AA)
    for index, point in enumerate(points):
        radius = 6 if index in TIP_INDEX.values() else 4
        cv2.circle(frame, tuple(map(int, point)), radius, color, -1, cv2.LINE_AA)
        if show_details:
            draw_text(frame, str(index), (int(point[0]) + 5, int(point[1]) - 5), (220, 220, 220), 0.35, 1)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 7, (0, 255, 255), -1, cv2.LINE_AA)
    cv2.arrowedLine(frame, (cx, cy), (int(cx + vx * 0.12), int(cy + vy * 0.12)), (0, 255, 255), 2, cv2.LINE_AA)

    angles = finger_angles(points)
    fingers = finger_states(points, label)
    gesture = classify_gesture(points, fingers)
    finger_text = " ".join(f"{name[0].upper()}:{'O' if fingers[name] else '-'}" for name in FINGER_NAMES)
    panel_x = min(max(x1, 8), max(width - 315, 8))
    panel_y = min(max(y1 - 112, 22), max(height - 125, 0))
    panel_w, panel_h = 305, 105
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
    draw_text(frame, f"{label} | {gesture}", (panel_x + 8, panel_y + 20), color, 0.52, 1)
    draw_text(frame, f"center=({cx},{cy})  speed={speed:6.1f}px/s", (panel_x + 8, panel_y + 41), (240, 240, 240), 0.43, 1)
    draw_text(frame, f"v=({vx:6.1f},{vy:6.1f})  arah={direction}", (panel_x + 8, panel_y + 60), (240, 240, 240), 0.43, 1)
    draw_text(frame, f"jari [T I M R P]: {finger_text}", (panel_x + 8, panel_y + 79), (240, 240, 240), 0.43, 1)
    draw_text(frame, f"sudut PIP I/M/R/P: {angles['index_pip_angle']:.0f}/{angles['middle_pip_angle']:.0f}/{angles['ring_pip_angle']:.0f}/{angles['pinky_pip_angle']:.0f}", (panel_x + 8, panel_y + 98), (240, 240, 240), 0.35, 1)


def csv_fieldnames() -> List[str]:
    fields = ["timestamp", "frame", "hand", "gesture", "direction", "center_x", "center_y",
              "velocity_x_px_s", "velocity_y_px_s", "speed_px_s", "normalized_speed",
              "pinch_distance_px", "thumb_open", "index_open", "middle_open", "ring_open", "pinky_open"]
    for i in range(21):
        fields.extend([f"lm{i}_x", f"lm{i}_y", f"lm{i}_z"])
    fields.extend([f"{name}_pip_angle" for name in ("index", "middle", "ring", "pinky")])
    fields.extend([f"{name}_dip_angle" for name in ("index", "middle", "ring", "pinky")])
    fields.append("thumb_ip_angle")
    fields.extend(["track_id", "video_time_s"])
    return fields


def make_csv_row(frame_number: int, label: str, normalized: List[Tuple[float, float, float]], points: List[Point],
                 motion: Tuple[float, float, float, str], width: int, height: int,
                 track_id: Optional[int] = None, video_time: Optional[float] = None) -> Dict[str, object]:
    vx, vy, speed, direction = motion
    states = finger_states(points, label)
    angles = finger_angles(points)
    center = palm_center(points)
    pinch = distance(points[THUMB_TIP], points[INDEX_TIP])
    row: Dict[str, object] = {
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "frame": frame_number,
        "hand": label,
        "gesture": classify_gesture(points, states),
        "direction": direction,
        "center_x": round(center[0], 3),
        "center_y": round(center[1], 3),
        "velocity_x_px_s": round(vx, 3),
        "velocity_y_px_s": round(vy, 3),
        "speed_px_s": round(speed, 3),
        "normalized_speed": round(speed / max(width, height), 6),
        "pinch_distance_px": round(pinch, 3),
        "track_id": track_id,
        "video_time_s": round(video_time, 6) if video_time is not None else "",
    }
    row.update({f"{name}_open": int(states[name]) for name in FINGER_NAMES})
    for i, (x, y, z) in enumerate(normalized):
        row[f"lm{i}_x"], row[f"lm{i}_y"], row[f"lm{i}_z"] = round(x, 6), round(y, 6), round(z, 6)
    row.update({key: round(value, 3) for key, value in angles.items()})
    return row


def close_windows() -> None:
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        # Beberapa build headless tidak menyediakan backend jendela.
        pass


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    if sys.version_info[:2] not in ((3, 11), (3, 12)):
        raise RuntimeError(
            "Interpreter Python tidak kompatibel untuk skrip ini. "
            "Skrip ini memerlukan Python 3.11 atau 3.12 karena memakai API legacy MediaPipe mp.solutions.hands.\n"
            f"Versi saat ini: {python_version} ({sys.executable})\n"
            "Gunakan venv yang sudah dibuat di folder proyek: "
            '"C:/Users/hp/OneDrive/Desktop/hand_detektor_Ai/.venv311/Scripts/python.exe" hand_motion_tracker.py\n'
            "Atau buat venv baru dengan:\n"
            "py -3.11 -m venv .venv311\n"
            ".venv311/Scripts/Activate.ps1\n"
            'python -m pip install "mediapipe==0.10.21" "opencv-contrib-python<4.12" "numpy<2" pygame\n'
            "Setelah itu jalankan skrip lagi."
        )
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError(
            'MediaPipe belum tersedia. Gunakan Python 3.11/3.12 dan jalankan:\n'
            'python -m pip install "mediapipe==0.10.21" "opencv-contrib-python<4.12" "numpy<2" pygame\n'
            'atau gunakan venv proyek: ".venv311/Scripts/python.exe" hand_motion_tracker.py'
        ) from exc
    if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "hands"):
        raise RuntimeError(
            "MediaPipe ini tidak menyediakan API legacy mp.solutions.hands. "
            "Penyebabnya paling sering adalah interpreter yang salah atau paket MediaPipe modern yang terpasang di venv yang salah.\n"
            f"Versi: {getattr(mp, '__version__', 'tidak diketahui')}; "
            f"lokasi: {getattr(mp, '__file__', 'tidak diketahui')}\n"
            "Gunakan Python 3.11/3.12 dan venv proyek (.venv311) atau install ulang dependensi tepat: "
            'python -m pip install "mediapipe==0.10.21" "opencv-contrib-python<4.12" "numpy<2" pygame\n'
            'Pastikan tidak ada modul lokal bernama mediapipe.py di folder kerja.'
        )
    pyvirtualcam = None
    if args.virtual_camera:
        try:
            import pyvirtualcam
        except ImportError as exc:
            raise RuntimeError(
                "Kamera virtual memerlukan OBS Studio dan python -m pip install pyvirtualcam."
            ) from exc

    tracker = HandTracker()
    shutdown = ShutdownCountdown()
    pointing_hold = PoseHold()
    pointing_enabled, shutdown_enabled = action_permissions(args)
    show_details = True
    frame_number = 0
    last_frame_time: Optional[float] = None
    last_countdown_report: Optional[int] = None
    window_name = "Detailed Hand Motion Tracker"
    virtual_camera = None
    try:
        # Seluruh akuisisi resource masuk cleanup, termasuk kegagalan saat startup.
        with ExitStack() as resources:
            if not args.no_display:
                resources.callback(close_windows)
            audio = GestureAudio()
            resources.callback(audio.close)
            audio.start(args)
            source = str(args.video) if args.video is not None else args.camera
            cap = cv2.VideoCapture(source)
            resources.callback(cap.release)
            if not cap.isOpened():
                raise RuntimeError(f"Sumber {source!r} tidak bisa dibuka. Periksa file, ID kamera, dan izin kamera.")
            if args.video is None:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
            video_clock = VideoClock(cap.get(cv2.CAP_PROP_FPS)) if args.video is not None else None
            csv_file = None
            writer = None
            if not args.no_csv:
                csv_path = Path(args.csv).expanduser().resolve()
                input_paths = [Path(__file__).resolve()]
                input_paths.extend(path.expanduser().resolve() for path in (args.video, args.audio, args.blur_audio)
                                   if path is not None)
                input_paths.extend(audio.tracks.values())
                if csv_path in input_paths:
                    raise ValueError("File CSV keluaran harus berbeda dari skrip, video, dan audio masukan.")
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                csv_file = resources.enter_context(csv_path.open("w", newline="", encoding="utf-8"))
                writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames())
                writer.writeheader()
                print(f"Data analisis disimpan ke: {csv_path}")
            print("Tracker aktif. Ctrl+C untuk keluar." if args.no_display else
                  "Tracker aktif. q/ESC keluar, x batal shutdown, c reset trail, s screenshot, d nomor titik.")
            if pointing_enabled:
                print("Tahan pose pointing dengan tangan yang sama selama 1,5 detik untuk keluar.")
            if args.video is not None:
                print("Mode rekaman: gesture pointing/shutdown dinonaktifkan; kecepatan memakai waktu video.")
            elif args.virtual_camera:
                print("Kamera virtual aktif. Pilih perangkat kamera virtual di aplikasi tujuan. "
                      "Gesture pointing/shutdown dinonaktifkan.")
            elif shutdown_enabled:
                print("Tahan jari tengah 1,5 detik untuk memulai hitungan 15 detik. Setelahnya Windows memaksa aplikasi tutup.")
            else:
                print("Gesture shutdown dinonaktifkan untuk sesi ini.")

            hands = resources.enter_context(mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=args.max_hands,
                model_complexity=1,
                min_detection_confidence=args.detection_confidence,
                min_tracking_confidence=args.tracking_confidence,
            ))
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    print("Sumber video selesai atau frame tidak bisa dibaca.")
                    break
                captured_at = time.monotonic()
                motion_time = (video_clock.timestamp(cap.get(cv2.CAP_PROP_POS_MSEC))
                               if video_clock is not None else captured_at)
                frame_number += 1
                if not args.no_mirror:
                    frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                result = hands.process(rgb)
                rgb.flags.writeable = True
                processed_at = time.monotonic()
                fps = 0.0 if last_frame_time is None else 1.0 / max(processed_at - last_frame_time, 1e-6)
                last_frame_time = processed_at
                detections = []
                hand_labels = result.multi_handedness or []
                for index, hand_landmarks in enumerate(result.multi_hand_landmarks or []):
                    label, confidence = "Unknown", 0.0
                    if index < len(hand_labels) and hand_labels[index].classification:
                        classification = hand_labels[index].classification[0]
                        label = correct_handedness(classification.label, mirrored_input=not args.no_mirror)
                        confidence = classification.score
                    normalized = normalized_landmarks(hand_landmarks)
                    points = pixel_landmarks(normalized, width, height)
                    fingers = finger_states(points, label)
                    gesture = classify_gesture(points, fingers)
                    detections.append((label, confidence, normalized, points, fingers, gesture))
                track_ids = tracker.associate(
                    [(entry[0], palm_center(entry[3])) for entry in detections], math.hypot(width, height))
                peace_pose_detected = False
                rock_pose_detected = False
                shutdown_ids = []
                pointing_ids = []
                for track_id, (label, confidence, normalized, points, fingers, gesture) in zip(track_ids, detections):
                    peace_pose_detected |= gesture == "peace"
                    rock_pose_detected |= gesture == "rock"
                    if is_shutdown_pose(points, fingers):
                        shutdown_ids.append(track_id)
                    if gesture == "pointing":
                        pointing_ids.append(track_id)
                    state = tracker.states[track_id]
                    motion = update_motion(state, palm_center(points), motion_time)
                    if not args.no_display or args.virtual_camera:
                        draw_hand(frame, points, f"#{track_id} {label} ({confidence:.0%})", (40, 220, 80),
                                  state, show_details, motion)
                    if writer is not None:
                        writer.writerow(make_csv_row(frame_number, label, normalized, points, motion, width, height,
                                                     track_id, motion_time if video_clock is not None else None))
                if not args.no_display or args.virtual_camera:
                    frame = apply_peace_blur(frame, peace_pose_detected)
                audio.update(peace_pose_detected, rock_pose_detected)
                action_time = time.monotonic()
                if shutdown_enabled and shutdown.observe(shutdown_ids, action_time):
                    print("Pose jari tengah stabil. Hitung mundur 15 detik dimulai; belum ada perintah Windows.")
                    last_countdown_report = None
                pointing_elapsed = pointing_hold.elapsed(pointing_ids if pointing_enabled else [], action_time)
                pointing_exit_remaining = None
                if pointing_elapsed is not None:
                    if pointing_elapsed >= POINTING_EXIT_HOLD_SECONDS:
                        print("Pose pointing stabil selama 1,5 detik. Aplikasi ditutup.")
                        break
                    pointing_exit_remaining = POINTING_EXIT_HOLD_SECONDS - pointing_elapsed
                remaining = shutdown.remaining(time.monotonic())
                if remaining is not None and remaining != last_countdown_report:
                    print(f"Shutdown dalam {remaining} detik. x batal; q/ESC atau Ctrl+C batal dan keluar.")
                    last_countdown_report = remaining
                if not args.no_display or args.virtual_camera:
                    draw_text(frame, f"Hands: {len(detections)}/{args.max_hands} | FPS proses: {fps:.1f}",
                              (12, 26), (0, 255, 255), 0.58, 1)
                    draw_text(frame, "q/ESC keluar | x batal shutdown | c reset | s screenshot | d detail",
                              (12, height - 14), (230, 230, 230), 0.48, 1)
                    if remaining is not None:
                        draw_text(frame, f"SHUTDOWN DALAM {remaining} DETIK - x / q / ESC BATAL",
                                  (12, 55), (0, 0, 255), 0.65, 2)
                    elif shutdown.command_sent:
                        draw_text(frame, "PERINTAH SHUTDOWN DIKIRIM KE WINDOWS", (12, 55), (0, 0, 255), 0.65, 2)
                    if pointing_exit_remaining is not None:
                        draw_text(frame, f"POINTING: tahan untuk keluar ({pointing_exit_remaining:.1f}s)",
                                  (12, 82), (0, 255, 255), 0.58, 2)
                if args.virtual_camera:
                    if virtual_camera is None:
                        virtual_camera = pyvirtualcam.Camera(width=width, height=height, fps=30,
                                                             fmt=pyvirtualcam.PixelFormat.BGR,
                                                             backend=args.virtual_camera_backend)
                        resources.callback(virtual_camera.close)
                        print(f"Mengirim video ke kamera virtual ({args.virtual_camera_backend}): "
                              f"{virtual_camera.device}")
                    virtual_camera.send(frame)
                    virtual_camera.sleep_until_next_frame()
                key = -1
                if not args.no_display:
                    cv2.imshow(window_name, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    try:
                        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                            break
                    except cv2.error:
                        # Tombol X jendela bisa menghancurkan window sebelum diperiksa.
                        break
                if key == ord("x"):
                    if shutdown.deadline is not None:
                        shutdown.cancel()
                        last_countdown_report = None
                        print("Hitung mundur dibatalkan. Lepaskan pose sebelum memulai lagi.")
                elif key == ord("c"):
                    for state in tracker.states.values():
                        reset_motion(state)
                    pointing_hold.since.clear()
                    shutdown.hold.since.clear()
                elif key == ord("d"):
                    show_details = not show_details
                elif key == ord("s"):
                    screenshot = Path(f"hand_tracker_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg").resolve()
                    try:
                        if not cv2.imwrite(str(screenshot), frame):
                            raise OSError("OpenCV tidak dapat menulis file")
                        print(f"Screenshot disimpan: {screenshot}")
                    except (OSError, cv2.error) as exc:
                        print(f"PERINGATAN: screenshot gagal disimpan: {exc}")
                # Tombol batal selalu diproses sebelum pengiriman shutdown.
                try:
                    if shutdown.deadline is not None and time.monotonic() >= shutdown.deadline and csv_file is not None:
                        csv_file.flush()
                    if shutdown.submit_if_due(time.monotonic()):
                        print("Perintah shutdown dikirim; tracker tetap berjalan sampai Windows menutupnya.")
                except RuntimeError as exc:
                    print(f"ERROR: {exc} Tracker tetap berjalan. Lepaskan pose sebelum mencoba lagi.")
    except KeyboardInterrupt:
        print("Tracker dihentikan dengan Ctrl+C.")
    finally:
        if shutdown.deadline is not None:
            shutdown.cancel()
            print("Hitung mundur lokal dibatalkan; tidak ada perintah shutdown yang dikirim ke Windows.")


if __name__ == "__main__":
    main()
