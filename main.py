"""
Real-time Hand VFX System with Neon Effects
Webcam application with glowing neon effects around hand movements.

Dependencies: OpenCV, MediaPipe, NumPy
"""

import cv2
import mediapipe as mp
import numpy as np
import random
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Configuration class for all visual effects parameters"""

    # Visual Effects
    glow_intensity: int = 6          # Number of glow layers (kept small to avoid fat bars)
    line_thickness: int = 2
    smoothing_factor: float = 0.3

    # Performance
    max_hands: int = 2
    detection_confidence: float = 0.5
    tracking_confidence: float = 0.5

    # Effects toggles
    enable_particles: bool = False
    enable_trails: bool = False
    enable_aura: bool = True
    enable_fingertip_pulse: bool = True
    enable_beam: bool = True

    # Cinematic VFX
    cinematic_darkening: float = 0.25
    bloom_intensity: float = 0.3
    pulse_speed: float = 3.0

    # Beam effect
    beam_fade_speed: float = 4.0

    # Neon line blend weight (higher = more visible lines)
    beam_blend_alpha: float = 0.85

    # Fingertip Join Effect
    join_effect_distance_threshold: float = 60.0
    join_effect_intensity: float = 1.0
    join_effect_max_size: int = 40
    join_effect_fade_speed: float = 8.0

    # Magic Shield Effect
    enable_shield: bool = True
    shield_radius: int = 120
    shield_rotation_speed: float = 2.0
    shield_glow_intensity: float = 0.6
    shield_color: Tuple[int, int, int] = (0, 140, 255)  # BGR orange

    # Skeleton glow falloff (lower = thinner/softer outer glow)
    skeleton_glow_alpha: float = 0.35

    # Aura enable/intensity
    aura_alpha: float = 0.08

    # Colors (BGR format for OpenCV)
    colors: dict = field(default_factory=lambda: {
        'primary':   (0, 255, 255),      # Cyan
        'secondary': (255, 0, 255),      # Magenta
        'accent':    (255, 255, 0),      # Yellow
        'energy':    (0, 255, 0),        # Green
        'purple':    (128, 0, 255),      # Purple
        'pink':      (255, 105, 180),    # Pink

        # Beam core colors
        'beam_core':      (255, 255, 255),
        'beam_highlight': (200, 220, 255),

        # Per-finger neon colors (BGR)
        'thumb_line':  (20, 80, 255),    # Orange-red
        'index_line':  (255, 255, 0),    # Cyan
        'middle_line': (220, 0, 255),    # Magenta
        'ring_line':   (255, 60, 140),   # Purple-blue
        'pinky_line':  (0, 220, 255),    # Yellow-green
    })

# ============================================================================
# HAND TRACKING
# ============================================================================

class HandTracker:
    """Handles hand detection using MediaPipe"""

    def __init__(self, config: Config):
        self.config = config
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config.max_hands,
            min_detection_confidence=config.detection_confidence,
            min_tracking_confidence=config.tracking_confidence,
            model_complexity=1
        )

        # Hand skeleton connections
        self.connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),        # Index
            (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
            (9, 13), (13, 14), (14, 15), (15, 16), # Ring
            (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
            (0, 17)                                 # Palm base
        ]

    def detect_hands(self, frame: np.ndarray) -> Optional[List[List[Tuple[int, int]]]]:
        """Detect hands and return pixel landmark positions"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if not results.multi_hand_landmarks:
            return None

        h, w = frame.shape[:2]
        all_landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                x = int(lm.x * w)
                y = int(lm.y * h)
                landmarks.append((x, y))
            all_landmarks.append(landmarks)

        return all_landmarks

# ============================================================================
# NEON RENDERING
# ============================================================================

class NeonRenderer:
    """Handles neon skeleton and aura rendering"""

    def __init__(self, config: Config):
        self.config = config
        self.pulse_time = 0.0
        self.connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)
        ]

    def update(self, dt: float):
        self.pulse_time += dt

    def draw_neon_line_on_overlay(self, overlay: np.ndarray,
                                   pt1: Tuple[int, int], pt2: Tuple[int, int],
                                   color: Tuple[int, int, int],
                                   base_thickness: int = 2):
        """
        Draw a neon glow line onto an overlay image.
        Uses only colored layers — no black, no dark outlines.
        Outer layers are dim, inner core is bright.
        """
        n = self.config.glow_intensity  # e.g. 6

        for i in range(n, 0, -1):
            # i=n is outermost, i=1 is innermost core
            thickness = base_thickness + (i - 1) * 2
            # Alpha: very faint outer, solid bright inner
            alpha = self.config.skeleton_glow_alpha * (1.0 - (i - 1) / n)
            alpha = max(0.0, min(1.0, alpha))
            if i == 1:
                alpha = 0.9  # Core line: nearly full brightness

            glow_color = tuple(int(c * alpha) for c in color)
            cv2.line(overlay, pt1, pt2, glow_color, thickness, cv2.LINE_AA)

    def draw_hand_skeleton(self, frame: np.ndarray,
                            landmarks: List[Tuple[int, int]],
                            color: Tuple[int, int, int]):
        """
        Draw hand skeleton using overlay blending so lines look glowing,
        not painted. Blends onto frame in-place.
        """
        h, w = frame.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)

        for c0, c1 in self.connections:
            if c0 < len(landmarks) and c1 < len(landmarks):
                pt1 = (int(landmarks[c0][0]), int(landmarks[c0][1]))
                pt2 = (int(landmarks[c1][0]), int(landmarks[c1][1]))
                self.draw_neon_line_on_overlay(overlay, pt1, pt2, color,
                                               self.config.line_thickness)

        # Blend skeleton overlay onto frame: additive-style blending
        # cv2.add clamps at 255 which gives the glow-on-dark look
        cv2.add(frame, overlay, dst=frame)

    def draw_hand_aura(self, frame: np.ndarray,
                       landmarks: List[Tuple[int, int]],
                       color: Tuple[int, int, int]):
        """
        Draw a soft pulsing aura ring around the palm center.
        Uses overlay blending — no dark blobs.
        """
        if not self.config.enable_aura or len(landmarks) < 21:
            return

        palm_indices = [0, 5, 9, 13, 17]
        palm_x = int(sum(landmarks[i][0] for i in palm_indices) / len(palm_indices))
        palm_y = int(sum(landmarks[i][1] for i in palm_indices) / len(palm_indices))

        pulse = math.sin(self.pulse_time * self.config.pulse_speed) * 0.15 + 1.0
        base_radius = int(35 * pulse)

        h, w = frame.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)

        # Draw soft rings — only on overlay, blended additively
        num_rings = 5
        for i in range(num_rings):
            radius = base_radius + i * 7
            # Fades out toward outer rings
            alpha = self.config.aura_alpha * (1.0 - i / num_rings)
            ring_color = tuple(int(c * alpha) for c in color)
            cv2.circle(overlay, (palm_x, palm_y), radius, ring_color, 2, cv2.LINE_AA)

        cv2.add(frame, overlay, dst=frame)

# ============================================================================
# BEAM / FINGERTIP CONNECTION EFFECT
# ============================================================================

class BeamEffect:
    """Renders RGB neon finger-to-finger connection lines between two hands"""

    def __init__(self, config: Config):
        self.config = config
        self.beam_alpha = 0.0
        self.target_alpha = 0.0
        self.time = 0.0
        self.hand1_landmarks: Optional[List[Tuple[int, int]]] = None
        self.hand2_landmarks: Optional[List[Tuple[int, int]]] = None

        # Tracking alpha for each finger's join effect
        self.join_alphas = {idx: 0.0 for idx in [4, 8, 12, 16, 20]}

        # Fingertip index → color
        self.finger_colors = {
            4:  config.colors['thumb_line'],
            8:  config.colors['index_line'],
            12: config.colors['middle_line'],
            16: config.colors['ring_line'],
            20: config.colors['pinky_line'],
        }

    def update(self, dt: float, landmarks_list):
        """Update fade state and store landmark data"""
        self.time += dt

        current_hand_count = len(landmarks_list) if landmarks_list else 0
        self.target_alpha = 1.0 if current_hand_count >= 2 else 0.0

        alpha_diff = self.target_alpha - self.beam_alpha
        self.beam_alpha += alpha_diff * dt * self.config.beam_fade_speed
        self.beam_alpha = max(0.0, min(1.0, self.beam_alpha))

        if landmarks_list and len(landmarks_list) >= 2:
            self.hand1_landmarks = landmarks_list[0]
            self.hand2_landmarks = landmarks_list[1]
            
            # Update join effect alphas based on distance
            hand1 = self.hand1_landmarks
            hand2 = self.hand2_landmarks
            for idx in self.join_alphas.keys():
                if idx < len(hand1) and idx < len(hand2):
                    x1, y1 = hand1[idx]
                    x2, y2 = hand2[idx]
                    dist = math.hypot(x2 - x1, y2 - y1)
                    
                    target = 1.0 if dist < self.config.join_effect_distance_threshold else 0.0
                    j_diff = target - self.join_alphas[idx]
                    new_alpha = self.join_alphas[idx] + j_diff * dt * self.config.join_effect_fade_speed
                    self.join_alphas[idx] = max(0.0, min(1.0, new_alpha))
                else:
                    self._fade_out_join_alpha(idx, dt)
        else:
            self.hand1_landmarks = None
            self.hand2_landmarks = None
            for idx in self.join_alphas.keys():
                self._fade_out_join_alpha(idx, dt)

    def _fade_out_join_alpha(self, idx: int, dt: float):
        j_diff = 0.0 - self.join_alphas[idx]
        new_alpha = self.join_alphas[idx] + j_diff * dt * self.config.join_effect_fade_speed
        self.join_alphas[idx] = max(0.0, min(1.0, new_alpha))

    def _draw_neon_line(self, overlay: np.ndarray,
                        pt1: Tuple[int, int], pt2: Tuple[int, int],
                        color: Tuple[int, int, int]):
        """
        Draw a single neon fingertip connection line onto overlay.
        Multiple passes with decreasing thickness for soft glow.
        No black border — only colored layers.
        """
        # Outer soft glow layers - upgraded for richer, more premium spread
        glow_passes = [
            (40, 0.06),   # furthest spread, very soft bloom
            (28, 0.12),   # wide halo
            (18, 0.20),   # mid halo
            (10, 0.40),   # inner bright halo
            (5,  0.75),   # thick core
            (2,  1.00),   # intense center
        ]
        for thickness, alpha_mult in glow_passes:
            layer_color = tuple(int(c * alpha_mult) for c in color)
            cv2.line(overlay, pt1, pt2, layer_color, thickness, cv2.LINE_AA)

    def _draw_join_effect(self, overlay: np.ndarray, center: Tuple[int, int], color: Tuple[int, int, int], alpha: float):
        """
        Draw a concentrated energy orb/burst at the contact area.
        Combines a pulsing bright core with a delicate magic cross-flare.
        """
        intensity = self.config.join_effect_intensity * alpha
        base_size = self.config.join_effect_max_size
        
        # Orb pulse dynamic sizing (fast subtle pulsing jitter)
        pulse = math.sin(self.time * 20.0) * 0.1 + 1.0
        size = int(base_size * pulse * alpha)
        
        if size <= 0:
            return
            
        # Energy burst glowing layers
        passes = [
            (size, 0.10, color),                 # faint far glow
            (int(size * 0.65), 0.35, color),     # bright corona
            (int(size * 0.35), 0.70, color),     # intense primary core
            (int(size * 0.15) + 1, 1.0, (255, 255, 255)) # tiny pure white center spark
        ]
        
        for radius, a_mult, c in passes:
            if radius <= 0:
                continue
            effective_a = min(1.0, a_mult * intensity)
            layer_color = tuple(int(ch * effective_a) for ch in c)
            cv2.circle(overlay, center, radius, layer_color, -1, cv2.LINE_AA)
            
        # Draw a sleek magical 4-point star flash (lens flare style)
        star_len = int(size * 1.8)
        if star_len > 0:
            star_color = tuple(int(ch * 0.5 * intensity) for ch in color)
            # Horizontal flare
            cv2.line(overlay, (center[0] - star_len, center[1]), (center[0] + star_len, center[1]), star_color, 2, cv2.LINE_AA)
            cv2.line(overlay, (center[0] - int(star_len*0.5), center[1]), (center[0] + int(star_len*0.5), center[1]), (255,255,255), 1, cv2.LINE_AA)
            # Vertical flare
            cv2.line(overlay, (center[0], center[1] - star_len), (center[0], center[1] + star_len), star_color, 2, cv2.LINE_AA)
            cv2.line(overlay, (center[0], center[1] - int(star_len*0.5)), (center[0], center[1] + int(star_len*0.5)), (255,255,255), 1, cv2.LINE_AA)

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw fingertip-to-fingertip neon connections and join effects.
        Returns the modified frame (IMPORTANT: caller must use return value).
        """
        if self.beam_alpha <= 0.01:
            return frame
        if self.hand1_landmarks is None or self.hand2_landmarks is None:
            return frame

        h, w = frame.shape[:2]
        
        # Overlay for connection lines
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Dedicated overlay for join effects (for 100% additive sparks)
        join_overlay = np.zeros((h, w, 3), dtype=np.uint8)
        has_join_effect = False

        fingertip_indices = [4, 8, 12, 16, 20]
        hand1 = self.hand1_landmarks
        hand2 = self.hand2_landmarks

        for idx in fingertip_indices:
            if idx < len(hand1) and idx < len(hand2):
                pt1 = (int(hand1[idx][0]), int(hand1[idx][1]))
                pt2 = (int(hand2[idx][0]), int(hand2[idx][1]))
                color = self.finger_colors.get(idx, self.config.colors['beam_core'])
                
                # Render beam neon line
                self._draw_neon_line(overlay, pt1, pt2, color)
                
                # Render join effect if fingers are close
                join_alpha = self.join_alphas.get(idx, 0.0)
                if join_alpha > 0.01:
                    has_join_effect = True
                    mx = int((pt1[0] + pt2[0]) / 2)
                    my = int((pt1[1] + pt2[1]) / 2)
                    self._draw_join_effect(join_overlay, (mx, my), color, join_alpha)

        # Blend connection lines softly
        # beam_blend_alpha controls how visible the lines are
        effective_alpha = self.config.beam_blend_alpha * self.beam_alpha
        cv2.addWeighted(overlay, effective_alpha, frame, 1.0, 0, dst=frame)
        
        # Add join effects directly on top (bright additive blend without muting)
        if has_join_effect:
            cv2.add(frame, join_overlay, dst=frame)

        return frame

# ============================================================================
# MAGIC SHIELD EFFECT
# ============================================================================

class MagicShieldEffect:
    """Dr Strange style magical circular shield effect"""

    def __init__(self, config: Config):
        self.config = config
        self.time = 0.0
        self.current_alpha = 0.0
        self.fade_speed = 8.0
        self.last_center = None

    def update(self, dt: float, landmarks_list):
        self.time += dt
        
        target_alpha = 0.0
        
        # Activate ONLY when exactly one hand is detected
        if landmarks_list and len(landmarks_list) == 1:
            landmarks = landmarks_list[0]
            
            # Simple open palm gesture check:
            # Check if fingertips are further from wrist than PIP joints
            wrist = (landmarks[0][0], landmarks[0][1])
            is_open = True
            for tip_idx, pip_idx in zip([8, 12, 16, 20], [6, 10, 14, 18]):
                tip = (landmarks[tip_idx][0], landmarks[tip_idx][1])
                pip = (landmarks[pip_idx][0], landmarks[pip_idx][1])
                dist_tip = math.hypot(tip[0] - wrist[0], tip[1] - wrist[1])
                dist_pip = math.hypot(pip[0] - wrist[0], pip[1] - wrist[1])
                if dist_tip < dist_pip:
                    is_open = False
                    break
                    
            if is_open:
                target_alpha = 1.0
                # Update last known center safely
                palm_indices = [0, 5, 9, 13, 17]
                cx = int(sum(landmarks[i][0] for i in palm_indices) / len(palm_indices))
                cy = int(sum(landmarks[i][1] for i in palm_indices) / len(palm_indices))
                self.last_center = (cx, cy)
                
        alpha_diff = target_alpha - self.current_alpha
        self.current_alpha += alpha_diff * dt * self.fade_speed
        self.current_alpha = max(0.0, min(1.0, self.current_alpha))

    def draw(self, frame: np.ndarray) -> np.ndarray:
        if not self.config.enable_shield or self.current_alpha <= 0.01:
            return frame
        if self.last_center is None:
            return frame

        h, w = frame.shape[:2]
        shield_overlay = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Mild flicker on glow intensity
        flicker = 1.0 + math.sin(self.time * 30.0) * 0.05
        intensity = self.config.shield_glow_intensity * self.current_alpha * flicker
        
        # Subtle scale pulsing
        pulse = math.sin(self.time * 5.0) * 0.02 + 1.0
        base_radius = int(self.config.shield_radius * pulse)
        
        color = self.config.shield_color
        
        angle_outer = self.time * self.config.shield_rotation_speed
        angle_inner = -self.time * self.config.shield_rotation_speed * 1.5
        
        self._draw_magic_circle(shield_overlay, self.last_center, base_radius, angle_outer, angle_inner, color, intensity)
        
        cv2.add(frame, shield_overlay, dst=frame)
        return frame

    def _draw_magic_circle(self, overlay: np.ndarray, center, radius, angle_outer, angle_inner, color, intensity):
        """Draw circular magic shield elements onto overlay with neon glow technique"""
        # Multi-layer drawing for neon glow
        thick_passes = [
            (24, 0.05),
            (16, 0.15),
            (8,  0.40),
            (3,  0.80),
            (1,  1.0)
        ]
        
        cx, cy = center
        
        for thickness, alpha in thick_passes:
            layer_alpha = alpha * intensity
            if layer_alpha <= 0: continue
            
            # gradient color effect: core is closer to white/bright yellow
            if thickness <= 3:
                r_c = min(255, color[0] + 100)
                g_c = min(255, color[1] + 100)
                b_c = min(255, color[2] + 100)
                c = (r_c, g_c, b_c)
            else:
                c = color
                
            layer_color = tuple(int(ch * layer_alpha) for ch in c)
            
            # 1. Main Outer Ring
            cv2.circle(overlay, center, radius, layer_color, thickness, cv2.LINE_AA)
            cv2.circle(overlay, center, max(1, radius - 15), layer_color, max(1, thickness - 1), cv2.LINE_AA)
            
            # 2. Outer Octagon
            if radius > 15:
                pts_oct = []
                for i in range(8):
                    theta = angle_outer + i * (math.pi / 4)
                    x = int(cx + (radius - 15) * math.cos(theta))
                    y = int(cy + (radius - 15) * math.sin(theta))
                    pts_oct.append((x, y))
                for i in range(8):
                    cv2.line(overlay, pts_oct[i], pts_oct[(i + 1) % 8], layer_color, max(1, thickness-1), cv2.LINE_AA)
            
            # 3. Inner Rotating Squares (giving that layered look)
            inner_r = max(5, radius - 45)
            pts_sq = []
            for i in range(4):
                theta = angle_inner + i * (math.pi / 2)
                x = int(cx + inner_r * math.cos(theta))
                y = int(cy + inner_r * math.sin(theta))
                pts_sq.append((x, y))
            for i in range(4):
                cv2.line(overlay, pts_sq[i], pts_sq[(i + 1) % 4], layer_color, thickness, cv2.LINE_AA)
                
            # 4. Connecting Radial Lines from Inner to Outer
            for i in range(12):
                theta = angle_outer + i * (math.pi / 6)
                x1 = int(cx + inner_r * math.cos(theta))
                y1 = int(cy + inner_r * math.sin(theta))
                x2 = int(cx + max(1, radius - 15) * math.cos(theta))
                y2 = int(cy + max(1, radius - 15) * math.sin(theta))
                cv2.line(overlay, (x1,y1), (x2,y2), layer_color, max(1, thickness-1), cv2.LINE_AA)
                
            # 5. Glowing Core (soft blend orb + small dot)
            core_r = int(25 * (1.0 + 0.1 * math.sin(self.time * 10.0)))
            if thickness > 3:
                cv2.circle(overlay, center, core_r, layer_color, -1, cv2.LINE_AA)
            elif thickness == 1:
                cv2.circle(overlay, center, 8, layer_color, -1, cv2.LINE_AA)
                
            # 6. Outer Runes/Arcs
            # Drawing disjoint arcs via cv2.ellipse
            arc_radius = int(radius + 20)
            axes = (arc_radius, arc_radius)
            for i in range(4):
                theta_mid = angle_inner * 1.5 + i * (math.pi / 2)
                start_rad = theta_mid - 0.2
                end_rad = theta_mid + 0.2
                cv2.ellipse(overlay, center, axes, 0, math.degrees(start_rad), math.degrees(end_rad), layer_color, thickness, cv2.LINE_AA)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

class HandVFXApp:
    """Main application class"""

    def __init__(self):
        self.config = Config()
        self.hand_tracker = HandTracker(self.config)
        self.neon_renderer = NeonRenderer(self.config)
        self.beam_effect = BeamEffect(self.config)
        self.shield_effect = MagicShieldEffect(self.config)
        self.last_time = time.time()
        self.frame_count = 0

        # Camera init with macOS fallback
        print("[VFX] Opening webcam...")
        self.cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        if not self.cap.isOpened():
            print("[VFX] CAP_AVFOUNDATION failed, trying default backend...")
            self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            print("[VFX] ERROR: Could not open webcam on index 0, trying index 1...")
            self.cap = cv2.VideoCapture(1)

        if not self.cap.isOpened():
            print("[VFX] ERROR: No webcam found. Exiting.")
            self.cap = None
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print("[VFX] Webcam opened successfully.")

        cv2.namedWindow("Hand VFX System", cv2.WINDOW_NORMAL)
        print("[VFX] Window created.")

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame and return it with VFX applied"""
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time

        self.neon_renderer.update(dt)

        landmarks_list = self.hand_tracker.detect_hands(frame)

        if landmarks_list:
            hand_colors = [
                self.config.colors['primary'],
                self.config.colors['secondary'],
            ]
            for hand_idx, landmarks in enumerate(landmarks_list):
                color = hand_colors[hand_idx % len(hand_colors)]
                self.neon_renderer.draw_hand_skeleton(frame, landmarks, color)
                self.neon_renderer.draw_hand_aura(frame, landmarks, color)

        # Update and draw beam connections
        if self.config.enable_beam:
            self.beam_effect.update(dt, landmarks_list)
            frame = self.beam_effect.draw(frame)  # MUST use returned frame

        # Update and draw magic shield
        self.shield_effect.update(dt, landmarks_list)
        frame = self.shield_effect.draw(frame)

        # Cinematic tint overlay
        if self.config.cinematic_darkening > 0:
            tint = np.full_like(frame, (10, 5, 20), dtype=np.uint8)
            cv2.addWeighted(frame, 1.0 - self.config.cinematic_darkening,
                            tint, self.config.cinematic_darkening, 0, dst=frame)

        # Subtle bloom
        if self.config.bloom_intensity > 0:
            bloom = cv2.GaussianBlur(frame, (21, 21), 0)
            cv2.addWeighted(frame, 1.0, bloom, self.config.bloom_intensity, 0, dst=frame)

        self.draw_info(frame)
        return frame

    def draw_info(self, frame: np.ndarray):
        """Draw HUD text"""
        lines = [
            "Hand VFX  |  q: quit",
            f"Aura: {'ON' if self.config.enable_aura else 'OFF'} (a)  "
            f"Beam: {'ON' if self.config.enable_beam else 'OFF'} (b)  "
            f"Shield: {'ON' if self.config.enable_shield else 'OFF'} (s)",
        ]
        y = 28
        for text in lines:
            cv2.putText(frame, text, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
            y += 24

    def handle_key(self, key: int):
        """Handle keyboard input"""
        if key == ord('a'):
            self.config.enable_aura = not self.config.enable_aura
            print(f"[VFX] Aura: {'ON' if self.config.enable_aura else 'OFF'}")
        elif key == ord('b'):
            self.config.enable_beam = not self.config.enable_beam
            print(f"[VFX] Beam: {'ON' if self.config.enable_beam else 'OFF'}")
        elif key == ord('s'):
            self.config.enable_shield = not self.config.enable_shield
            print(f"[VFX] Shield: {'ON' if self.config.enable_shield else 'OFF'}")

    def run(self):
        """Main application loop"""
        if self.cap is None:
            print("[VFX] Cannot run — no camera available.")
            return

        print("[VFX] Starting. Controls: q=quit  a=aura  b=beam  s=shield")

        # Validate first frame
        ret, frame = self.cap.read()
        if not ret or frame is None:
            print("[VFX] ERROR: Could not read first frame.")
            self.cap.release()
            return
        print("[VFX] First frame received. Entering main loop.")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    print("[VFX] WARNING: Frame read failed, retrying...")
                    continue

                frame = cv2.flip(frame, 1)
                processed = self.process_frame(frame)

                cv2.imshow("Hand VFX System", processed)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[VFX] Quit key pressed.")
                    break
                self.handle_key(key)

                self.frame_count += 1

        except KeyboardInterrupt:
            print("[VFX] Interrupted by user.")

        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("[VFX] Application closed.")


def main():
    app = HandVFXApp()
    app.run()


if __name__ == "__main__":
    main()
