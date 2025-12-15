import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import math
import time

class FallDetectionSystem:
    def __init__(self, yolo_model_path='yolov8n.pt', use_yolo_pose=True):
        print("Initializing Fall Detection System...")
        self.yolo_model = YOLO(yolo_model_path)
        self.use_yolo_pose = use_yolo_pose
       
        if use_yolo_pose:
            print("Using YOLOv8-Pose for keypoint detection...")
            self.pose_model = YOLO('yolov8n-pose.pt')
       
        self.fall_timers = {}
        print("System initialized successfully!\n")
       
    def detect_persons(self, frame):
        results = self.yolo_model(frame, classes=[0], verbose=False)
        persons = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                if confidence > 0.5:
                    persons.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': confidence
                    })
        return persons

    def get_bottom_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return np.array([(x1 + x2) / 2, y2])

    def construct_perpendicular_lines(self, bbox, person_height=None):
        bottom_center = self.get_bottom_center(bbox)
        line_length = person_height if person_height else (bbox[3] - bbox[1])
        half_length = line_length / 2
        horizontal_line = {
            'start': np.array([bottom_center[0] - half_length, bottom_center[1]]),
            'end': np.array([bottom_center[0] + half_length, bottom_center[1]]),
            'direction': np.array([1, 0, 0])
        }
        return {'center': bottom_center, 'horizontal': horizontal_line, 'length': line_length}

    def draw_perpendicular_lines(self, frame, lines, color=(0, 200, 200), thickness=2):
        h_start = tuple(lines['horizontal']['start'].astype(int))
        h_end = tuple(lines['horizontal']['end'].astype(int))
        cv2.line(frame, h_start, h_end, color, thickness)
        center = tuple(lines['center'].astype(int))
        cv2.circle(frame, center, 5, (255, 0, 0), -1)
        label = f"H-Line: {lines['length']:.0f}px"
        cv2.putText(frame, label, (int(lines['center'][0] - 50), int(lines['center'][1] - 20)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        return frame

    def get_pose_landmarks_yolo_full_frame(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        results = self.pose_model(frame, verbose=False)
        if len(results) == 0 or results[0].keypoints is None:
            return None
        best_match = None
        best_overlap = 0
        for person_kp in results[0].keypoints.data:
            if len(person_kp) == 0: continue
            overlap_count = sum(1 for kp in person_kp if len(kp) >= 2 and x1 <= kp[0] <= x2 and y1 <= kp[1] <= y2)
            if overlap_count > best_overlap:
                best_overlap = overlap_count
                best_match = person_kp
        if best_match is None: return None
        landmarks = {}
        for idx, kp in enumerate(best_match):
            if len(kp) >= 3:
                landmarks[idx] = {'x': float(kp[0]), 'y': float(kp[1]), 'z': 0, 'visibility': float(kp[2])}
        return landmarks if landmarks else None

    def get_key_points_yolo(self, landmarks, visibility_threshold=0.3):
        indices = [5,6,11,12,13,14,15,16]
        if not all(i in landmarks and landmarks[i]['visibility'] > visibility_threshold for i in indices):
            return None
        mid = lambda a, b: np.array([(landmarks[a]['x'] + landmarks[b]['x'])/2, (landmarks[a]['y'] + landmarks[b]['y'])/2, 0])
        return {
            'mid_shoulder': mid(5,6), 'mid_hip': mid(11,12),
            'mid_knee': mid(13,14), 'mid_ankle': mid(15,16),
            'all_landmarks': landmarks
        }

    def calculate_person_height(self, key_points):
        return np.linalg.norm(key_points['mid_shoulder'][:2] - key_points['mid_ankle'][:2])

    def project_vector_on_line(self, vector, line_direction):
        v2d = vector[:2]
        dir2d = line_direction[:2] / (np.linalg.norm(line_direction[:2]) + 1e-6)
        return abs(np.dot(v2d, dir2d))

    def determine_body_vector(self, key_points):
        mid_shoulder = key_points['mid_shoulder']
        mid_hip = key_points['mid_hip']
        mid_knee = key_points['mid_knee']
        mid_ankle = key_points['mid_ankle']
        angle = self.calculate_angle(mid_shoulder, mid_hip, mid_knee)
        if 85 <= angle <= 100:
            body_vector = mid_ankle - mid_hip
            ref = mid_hip
            vtype = "hip-ankle"
        else:
            adj = (mid_shoulder + mid_knee) / 2
            body_vector = mid_ankle - adj
            ref = adj
            vtype = "adjusted"
        return body_vector, ref, angle, vtype

    def calculate_angle(self, p1, p2, p3):
        v1 = p1 - p2
        v2 = p3 - p2
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos = np.clip(cos, -1, 1)
        return np.degrees(np.arccos(cos))

    def draw_all_landmarks(self, frame, landmarks):
        for idx, lm in landmarks.items():
            x, y = int(lm['x']), int(lm['y'])
            vis = lm['visibility']
            color = (0, 255, 0) if vis > 0.7 else (0, 255, 255) if vis > 0.5 else (0, 0, 255)
            cv2.circle(frame, (x, y), 3, color, -1)
        return frame

    def process_frame(self, frame, frame_number, fps):
        output_frame = frame.copy()
        alerts = []
        active_persons = set()

        # العتبات - تقدر تعدلها
        FALL_START_THRESHOLD = 0.35
        FALL_COMPLETE_THRESHOLD = 0.93

        persons = self.detect_persons(frame)

        for idx, person in enumerate(persons):
            bbox = person['bbox']
            x1, y1, x2, y2 = bbox
            active_persons.add(idx)

            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            id_text = f"Person {idx}"
            id_bg = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(output_frame, (x1, y1 - id_bg[1] - 10), (x1 + id_bg[0] + 10, y1), (0, 255, 0), -1)
            cv2.putText(output_frame, id_text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            landmarks = self.get_pose_landmarks_yolo_full_frame(frame, bbox)
            if not landmarks:
                cv2.putText(output_frame, "No pose detected", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                continue

            output_frame = self.draw_all_landmarks(output_frame, landmarks)
            key_points = self.get_key_points_yolo(landmarks)
            if not key_points:
                cv2.putText(output_frame, "Key points not visible", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                continue

            person_height = self.calculate_person_height(key_points)
            lines = self.construct_perpendicular_lines(bbox, person_height)
            output_frame = self.draw_perpendicular_lines(output_frame, lines)

            # رسم النقاط الرئيسية
            for name, point in key_points.items():
                if name != 'all_landmarks':
                    cv2.circle(output_frame, (int(point[0]), int(point[1])), 6, (255, 0, 255), -1)
                    cv2.putText(output_frame, name.split('_')[1][:3].upper(), (int(point[0]) + 8, int(point[1]) - 8),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

            body_vector, ref_point, angle, vtype = self.determine_body_vector(key_points)
            start_pt = ref_point[:2].astype(int)
            end_pt = (ref_point[:2] + body_vector[:2] * 1.5).astype(int)
            cv2.arrowedLine(output_frame, tuple(start_pt), tuple(end_pt), (255, 0, 255), 3, tipLength=0.3)

            proj_h = self.project_vector_on_line(body_vector, lines['horizontal']['direction'])
            body_len = np.linalg.norm(body_vector[:2]) + 1e-6
            normalized_proj_h = proj_h / body_len

            alert_triggered = normalized_proj_h > FALL_START_THRESHOLD
            fall_completed = normalized_proj_h >= FALL_COMPLETE_THRESHOLD

            if idx not in self.fall_timers:
                self.fall_timers[idx] = {
                    'state': 'standing',      
                    'start_time': None,       
                    'start_frame': None,
                    'duration_sec': None,
                    'duration_frames': None
                }

            t = self.fall_timers[idx]
            now = time.perf_counter()

            if t['state'] == 'standing' and alert_triggered:
                t['state'] = 'falling'
                t['start_time'] = now
                t['start_frame'] = frame_number
                t['duration_sec'] = t['duration_frames'] = None
                print(f"\nPerson {idx} Fall starts - Frame {frame_number}")

            elif t['state'] == 'falling':
                if fall_completed:
                    t['state'] = 'fell'
                    t['duration_sec'] = now - t['start_time']
                    t['duration_frames'] = frame_number - t['start_frame']
                    print(f"Person {idx} fall happend in {t['duration_sec']:.3f}s ({t['duration_frames']} فريم)")

                elif not alert_triggered:
                    print(f"Person {idx} has fallen")
                    t['state'] = 'standing'
                    t['start_time'] = t['start_frame'] = t['duration_sec'] = t['duration_frames'] = None

            elif t['state'] == 'fell' and not alert_triggered:
                t['state'] = 'standing'
                t['start_time'] = t['start_frame'] = t['duration_sec'] = t['duration_frames'] = None

            if t['state'] == 'falling':
                elapsed_sec = now - t['start_time']
                elapsed_frames = frame_number - t['start_frame']
                status = f"Person {idx} - FALLING!"
                color = (0, 0, 255)
                info = f"Falling: {elapsed_sec:.3f}s ({elapsed_frames} frames)"

            elif t['state'] == 'fell':
                status = f"Person {idx} - FALL COMPLETE! ({t['duration_sec']:.3f}s)"
                color = (255, 0, 255)
                info = f"Duration: {t['duration_sec']:.3f}s ({t['duration_frames']} frames)"

            else:
                status = f"Person {idx}"
                color = (0, 255, 0)
                info = "Status: NORMAL"

            if t['state'] in ['falling', 'fell']:
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 3)
                txt_size = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(output_frame, (x1, y1 - txt_size[1] - 50), (x1 + txt_size[0] + 10, y1 - 40), color, -1)
                cv2.putText(output_frame, status, (x1 + 5, y1 - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            info_lines = [
                f"ID: {idx}", f"Frame: {frame_number}",
                f"Proj_H: {normalized_proj_h:.3f}", f"Type: {vtype}",
                info
            ]
            for i, line in enumerate(info_lines):
                cv2.putText(output_frame, line, (x1, y2 + 20 + i*20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if t['state'] in ['falling', 'fell']:
                counter_y = 30 + len([p for p in self.fall_timers.values() if p['state'] in ['falling', 'fell']]) * 50
                if t['state'] == 'falling':
                    txt = f"Person {idx} Falling: {now - t['start_time']:.3f}s"
                    bg_color = (0, 69, 255)
                else:
                    txt = f"Person {idx} Fell in: {t['duration_sec']:.3f}s"
                    bg_color = (0, 200, 0)

                txt_size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                cv2.rectangle(output_frame, (output_frame.shape[1] - txt_size[0] - 30, counter_y - 25),
                            (output_frame.shape[1] - 10, counter_y + 5), bg_color, -1)
                cv2.putText(output_frame, txt, (output_frame.shape[1] - txt_size[0] - 20, counter_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return output_frame, alerts

    def process_video(self, video_path, output_path=None, display=True, save_output=True):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video: {width}x{height} @ {fps} FPS, Total frames: {total_frames}")

        out = None
        if save_output:
            output_path = output_path or f"fall_detection_output_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        self.fall_timers = {} 
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1

            output_frame, _ = self.process_frame(frame, frame_count, fps)

            if out: out.write(output_frame)
            if display:
                cv2.putText(output_frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                if save_output:
                    cv2.circle(output_frame, (width - 30, 30), 8, (0, 0, 255), -1)
                    cv2.putText(output_frame, "REC", (width - 70, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow('Fall Detection System - Press Q to Quit', output_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        if out: out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = FallDetectionSystem(use_yolo_pose=True)
    video_path = "C:/Users/Kareem/Desktop/Human Fall Detection - Different Falling Postures and its Accuracy.mp4"
    
    detector.process_video(
        video_path=video_path,
        output_path="final_fall_detection.mp4",
        display=True,
        save_output=True
    )