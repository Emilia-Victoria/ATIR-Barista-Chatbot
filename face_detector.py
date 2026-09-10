import cv2
import mediapipe as mp
from collections import deque
import math
import threading
import time

class FaceDetector:
    """
    MAR (Mouth Aspect Ratio) and Head Position Detector
    
    Tracks:
    - Speech detection via MAR
    - Face visibility (selects nearest if multiple faces)
    - Head position (X, Y coordinates)
    
    Usage:
        detector = FaceDetector()
        detector.start()
        
        # Typical loop:
        is_speaking = detector.is_speaking()
        face_visible = detector.is_face_visible()
        head_x, head_y = detector.get_head_position()
        
        detector.stop()
    """
    
    def __init__(self, camera_index=0, speaking_threshold=0.04, window_size=15, speaking_history_seconds=1.0, fps_estimate=30):
        """
        Initialize the MAR detector
        
        Args:
            camera_index (int): Camera device index (usually 0)
            speaking_threshold (float): MAR threshold for speaking detection
            window_size (int): Number of frames to average MAR over
            speaking_history_seconds (float): How long to remember speaking activity (in seconds)
            fps_estimate (int): Estimated camera FPS for calculating speaking buffer size
        """
        # MediaPipe setup
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False, 
            max_num_faces=3,  # Allow multiple faces, we'll select the nearest
            refine_landmarks=False,  # Don't need iris landmarks anymore
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Mouth landmark indices (MediaPipe's 468-point model)
        self.TOP_LIP = [13]
        self.BOTTOM_LIP = [14]
        self.LEFT_MOUTH = [78]
        self.RIGHT_MOUTH = [308]
        
        # Head pose landmarks (nose tip and corners for position estimation)
        self.NOSE_TIP = [1]
        self.LEFT_FACE = [234]   # Left side of face
        self.RIGHT_FACE = [454]  # Right side of face
        self.CHIN = [175]
        self.FOREHEAD = [10]
        
        # Configuration
        self.camera_index = camera_index
        self.speaking_threshold = speaking_threshold
        self.window_size = window_size
        self.speaking_history_seconds = speaking_history_seconds
        self.fps_estimate = max(1, min(15, fps_estimate))
        
        self.speaking_buffer_size = max(1, int(self.fps_estimate * speaking_history_seconds))
        
        # Data buffers
        self.mar_buffer = deque(maxlen=window_size)
        self.speaking_buffer = deque(maxlen=self.speaking_buffer_size)  # Track speaking over time
        self.head_position_buffer_x = deque(maxlen=5)  # Smaller buffer for smoother head tracking
        self.head_position_buffer_y = deque(maxlen=5)
        
        # Thread-safe data storage
        self.lock = threading.Lock()
        self.current_mar = 0.0
        self.speaking = False
        self.face_visible = False
        self.head_x = 0.0  # Normalized coordinates (-1.0 to 1.0, left to right)
        self.head_y = 0.0  # Normalized coordinates (-1.0 to 1.0, up to down)
        self.face_size = 0.0  # For selecting nearest face
        
        # Threading controls
        self.thread = None
        self.running = False
        self.cap = None
    
    def euclidean_distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])
    
    def calculate_mar(self, landmarks, image_w, image_h):
        """Calculate Mouth Aspect Ratio (MAR)"""
        try:
            top = (
                landmarks[self.TOP_LIP[0]].x * image_w, 
                landmarks[self.TOP_LIP[0]].y * image_h
            )
            bottom = (
                landmarks[self.BOTTOM_LIP[0]].x * image_w, 
                landmarks[self.BOTTOM_LIP[0]].y * image_h
            )
            left = (
                landmarks[self.LEFT_MOUTH[0]].x * image_w, 
                landmarks[self.LEFT_MOUTH[0]].y * image_h
            )
            right = (
                landmarks[self.RIGHT_MOUTH[0]].x * image_w, 
                landmarks[self.RIGHT_MOUTH[0]].y * image_h
            )
            
            vertical = self.euclidean_distance(top, bottom)
            horizontal = self.euclidean_distance(left, right)
            
            mar = vertical / horizontal if horizontal > 0 else 0
            return mar
        except (IndexError, AttributeError):
            return 0.0
    
    def calculate_face_size(self, landmarks, image_w, image_h):
        """Calculate face size to determine nearest face"""
        try:
            left = landmarks[self.LEFT_FACE[0]]
            right = landmarks[self.RIGHT_FACE[0]]
            top = landmarks[self.FOREHEAD[0]]
            bottom = landmarks[self.CHIN[0]]
            
            width = abs(right.x - left.x) * image_w
            height = abs(bottom.y - top.y) * image_h
            
            return width * height  # Face area as proxy for distance (larger = closer)
        except (IndexError, AttributeError):
            return 0.0
    
    def calculate_head_position(self, landmarks, image_w, image_h):
        """Calculate normalized head position (-1 to 1 for both X and Y)"""
        try:
            # Use nose tip as main reference point
            nose = landmarks[self.NOSE_TIP[0]]
            
            # Convert to normalized coordinates
            # X: -1.0 = left edge, 0.0 = center, 1.0 = right edge
            head_x = (nose.x - 0.5) * 2.0
            
            # Y: -1.0 = top edge, 0.0 = center, 1.0 = bottom edge
            head_y = (nose.y - 0.5) * 2.0
            
            # Clamp to valid range
            head_x = max(-1.0, min(1.0, head_x))
            head_y = max(-1.0, min(1.0, head_y))
            
            return head_x, head_y
            
        except (IndexError, AttributeError):
            return 0.0, 0.0
    
    def select_nearest_face(self, multi_face_landmarks, image_w, image_h):
        """Select the nearest (largest) face from multiple detected faces"""
        if not multi_face_landmarks:
            return None
        
        if len(multi_face_landmarks) == 1:
            return multi_face_landmarks[0]
        
        # Calculate face size for each detected face
        largest_face = None
        largest_size = 0
        
        for face_landmarks in multi_face_landmarks:
            face_size = self.calculate_face_size(face_landmarks.landmark, image_w, image_h)
            if face_size > largest_size:
                largest_size = face_size
                largest_face = face_landmarks
        
        return largest_face
    
    def _capture_loop(self):
        """Main capture loop running in background thread"""
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            self.running = False
            return

        # A modest capture size and frame rate are sufficient for face landmarks.
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        capture_fps = self.fps_estimate
        self.cap.set(cv2.CAP_PROP_FPS, capture_fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print(f"MAR Detector started with camera {self.camera_index}")
        
        frame_interval = 1.0 / capture_fps
        next_process_time = 0.0

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            current_time = time.monotonic()
            if current_time < next_process_time:
                continue
            next_process_time = current_time + frame_interval
            
            # Flip horizontally and convert to RGB
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            
            # Face mesh detection
            results = self.face_mesh.process(rgb)
            image_h, image_w, _ = frame.shape
            
            with self.lock:
                if results.multi_face_landmarks:
                    # Select the nearest face
                    selected_face = self.select_nearest_face(results.multi_face_landmarks, image_w, image_h)
                    
                    if selected_face:
                        self.face_visible = True
                        
                        # Calculate MAR
                        mar = self.calculate_mar(selected_face.landmark, image_w, image_h)
                        self.mar_buffer.append(mar)
                        
                        # Check if this frame shows speaking (individual frame MAR above threshold)
                        frame_is_speaking = mar > self.speaking_threshold
                        self.speaking_buffer.append(frame_is_speaking)
                        
                        # Calculate head position
                        head_x, head_y = self.calculate_head_position(selected_face.landmark, image_w, image_h)
                        self.head_position_buffer_x.append(head_x)
                        self.head_position_buffer_y.append(head_y)
                        
                        # Update current values with averages
                        self.current_mar = sum(self.mar_buffer) / len(self.mar_buffer)
                        self.head_x = sum(self.head_position_buffer_x) / len(self.head_position_buffer_x)
                        self.head_y = sum(self.head_position_buffer_y) / len(self.head_position_buffer_y)
                        
                        # Update speaking status: True if ANY recent frame was speaking
                        self.speaking = any(self.speaking_buffer) if self.speaking_buffer else False
                        
                        # Store face size for reference
                        self.face_size = self.calculate_face_size(selected_face.landmark, image_w, image_h)
                    else:
                        self.face_visible = False
                        # Add False to speaking buffer when no face is detected
                        self.speaking_buffer.append(False)
                        self.speaking = any(self.speaking_buffer) if self.speaking_buffer else False
                else:
                    self.face_visible = False
                    # Add False to speaking buffer when no face is detected
                    self.speaking_buffer.append(False)
                    self.speaking = any(self.speaking_buffer) if self.speaking_buffer else False
            

        if self.cap:
            self.cap.release()
            self.cap = None
        print("MAR Detector stopped")
    
    def start(self):
        """Start the background detection thread"""
        if self.running:
            print("MAR Detector is already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        
        # Wait a moment for initialization
        time.sleep(0.5)
        
    def stop(self):
        """Stop the background detection thread"""
        if not self.running:
            return
        
        self.running = False
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        
        # Ensure camera is released
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def is_speaking(self):
        """Get current speaking status (thread-safe)
        
        Returns True if the person was speaking within the recent time window
        (speaking_history_seconds), even if they're not speaking at this exact moment.
        """
        with self.lock:
            return self.speaking
    
    def is_face_visible(self):
        """Check if a face is currently detected (thread-safe)"""
        with self.lock:
            return self.face_visible
    
    def get_head_position(self):
        """Get current head position as (x, y) tuple (thread-safe)
        
        Returns:
            tuple: (head_x, head_y) where both are in range -1.0 to 1.0
                   head_x: -1.0 = left, 0.0 = center, 1.0 = right
                   head_y: -1.0 = top, 0.0 = center, 1.0 = bottom
        """
        with self.lock:
            return self.head_x, self.head_y
    
    def get_mar(self):
        """Get current MAR value (thread-safe)"""
        with self.lock:
            return self.current_mar
    
    def get_face_size(self):
        """Get current face size (larger = closer to camera) (thread-safe)"""
        with self.lock:
            return self.face_size
    
    def get_status(self):
        """Get all current status information (thread-safe)"""
        with self.lock:
            return {
                'speaking': self.speaking,
                'face_visible': self.face_visible,
                'head_x': self.head_x,
                'head_y': self.head_y,
                'mar': self.current_mar,
                'face_size': self.face_size
            }
    
    def set_speaking_threshold(self, threshold):
        """Update speaking threshold (thread-safe)"""
        with self.lock:
            self.speaking_threshold = threshold
    
    def set_speaking_history(self, seconds):
        """Update how long to remember speaking activity (thread-safe)"""
        with self.lock:
            self.speaking_history_seconds = seconds
            new_buffer_size = max(1, int(self.fps_estimate * seconds))
            
            # Create new buffer with updated size, preserving recent data
            old_data = list(self.speaking_buffer)
            self.speaking_buffer = deque(old_data[-new_buffer_size:], maxlen=new_buffer_size)
            self.speaking_buffer_size = new_buffer_size
    
    def __enter__(self):
        """Context manager support"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup"""
        self.stop()


# Example usage and testing
if __name__ == "__main__":
    print("Testing MAR and Head Position Detector...")
    print("Move your head around and try speaking!")
    
    # Test the detector
    detector = FaceDetector()
    
    try:
        detector.start()
        
        print("Running for 30 seconds...")
        start_time = time.time()
        
        while time.time() - start_time < 30:
            status = detector.get_status()
            
            if status['face_visible']:
                print(f"\rMAR: {status['mar']:.3f} | "
                      f"Speaking: {'YES' if status['speaking'] else 'NO'} | "
                      f"Head: ({status['head_x']:+.2f}, {status['head_y']:+.2f}) | "
                      f"Size: {status['face_size']:.0f}", end='')
            else:
                print(f"\rNo face detected...", end='')
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        detector.stop()
        print("\nTest completed!")
