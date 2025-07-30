import cv2
import numpy as np
import sys
from ultralytics import YOLO
from collections import deque
import time

print("Starting YOLOv8 Ultra-Accurate & Stable Object Detection...")
print("Python version:", sys.version)

def load_yolov8():
    """Load YOLOv8 model - using larger model for better accuracy"""
    print("Loading YOLOv8 model...")
    try:
        # Try to load a larger model for better accuracy
        model = YOLO('yolov8s.pt')  # Small model - better than nano, faster than medium
        print("YOLOv8s model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading YOLOv8s model: {e}")
        print("Falling back to YOLOv8n model...")
        try:
            model = YOLO('yolov8n.pt')
            print("YOLOv8n model loaded successfully!")
            return model
        except Exception as e2:
            print(f"Failed to load YOLOv8 model: {e2}")
            return None

class UltraStableDetector:
    def __init__(self):
        self.detection_history = deque(maxlen=20)  # Longer history
        self.object_tracks = {}  # Track individual objects
        self.last_detection_time = 0
        self.detection_interval = 0.3  # Detect every 0.3 seconds
        self.min_confidence = 0.35  # Higher confidence for better accuracy
        self.stability_threshold = 0.4  # 40% consistency
        self.object_persistence = 2.0  # Keep objects visible for 2 seconds
        self.last_object_times = {}  # Track when each object was last seen
        
        # Class-specific confidence thresholds for better accuracy
        self.class_confidence_thresholds = {
            'cell phone': 0.4,  # Higher threshold for phones
            'phone': 0.4,
            'mobile phone': 0.4,
            'person': 0.3,  # Lower threshold for people
            'teddy bear': 0.5,  # Very high threshold for teddy bears
            'remote': 0.45,  # Higher threshold for remotes
            'remote control': 0.45,
            'pen': 0.4,  # Higher threshold for pens
            'pencil': 0.4,
            'book': 0.35,
            'cup': 0.35,
            'bottle': 0.35,
            'laptop': 0.4,
            'keyboard': 0.4,
            'mouse': 0.4,
        }
        
    def calculate_distance(self, bbox1, bbox2):
        """Calculate distance between two bounding boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        center1 = (x1 + w1//2, y1 + h1//2)
        center2 = (x2 + w2//2, y2 + h2//2)
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union between two bounding boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def filter_overlapping_detections(self, detections):
        """Remove overlapping detections, keeping the one with higher confidence"""
        if len(detections) <= 1:
            return detections
            
        filtered = []
        for i, det1 in enumerate(detections):
            keep = True
            for j, det2 in enumerate(detections):
                if i != j:
                    iou = self.calculate_iou(det1['bbox'], det2['bbox'])
                    if iou > 0.5:  # If overlap is more than 50%
                        # Keep the one with higher confidence
                        if det1['confidence'] < det2['confidence']:
                            keep = False
                            break
            if keep:
                filtered.append(det1)
        
        return filtered
    
    def apply_class_specific_thresholds(self, detections):
        """Apply different confidence thresholds for different classes"""
        filtered = []
        for det in detections:
            class_name = det['class_name'].lower()
            confidence = det['confidence']
            
            # Get threshold for this class, or use default
            threshold = self.class_confidence_thresholds.get(class_name, self.min_confidence)
            
            if confidence >= threshold:
                filtered.append(det)
        
        return filtered
    
    def get_stable_detections(self, current_detections):
        """Get ultra-stable detections with object tracking and accuracy improvements"""
        current_time = time.time()
        
        # Apply accuracy improvements
        if current_detections:
            # Filter by class-specific confidence thresholds
            current_detections = self.apply_class_specific_thresholds(current_detections)
            # Remove overlapping detections
            current_detections = self.filter_overlapping_detections(current_detections)
        
        # Add current detections to history
        self.detection_history.append(current_detections)
        
        # Update object tracks
        for det in current_detections:
            obj_name = det['class_name']
            bbox = det['bbox']
            
            if obj_name not in self.object_tracks:
                self.object_tracks[obj_name] = []
            
            # Find closest existing track
            min_distance = float('inf')
            best_match = None
            
            for track in self.object_tracks[obj_name]:
                distance = self.calculate_distance(bbox, track['bbox'])
                if distance < min_distance and distance < 100:  # Max 100px distance
                    min_distance = distance
                    best_match = track
            
            if best_match:
                # Update existing track with smoothed position
                alpha = 0.7  # Smoothing factor
                x, y, w, h = bbox
                tx, ty, tw, th = best_match['bbox']
                smoothed_bbox = (
                    int(alpha * x + (1-alpha) * tx),
                    int(alpha * y + (1-alpha) * ty),
                    int(alpha * w + (1-alpha) * tw),
                    int(alpha * h + (1-alpha) * th)
                )
                best_match['bbox'] = smoothed_bbox
                best_match['confidence'] = det['confidence']
                best_match['last_seen'] = current_time
            else:
                # Create new track
                self.object_tracks[obj_name].append({
                    'bbox': bbox,
                    'confidence': det['confidence'],
                    'last_seen': current_time
                })
            
            self.last_object_times[obj_name] = current_time
        
        # Get stable objects with persistence
        stable_objects = []
        for obj_name, tracks in self.object_tracks.items():
            if obj_name in self.last_object_times:
                time_since_seen = current_time - self.last_object_times[obj_name]
                
                # Keep object visible if seen recently or if it's consistently detected
                if time_since_seen < self.object_persistence:
                    # Get the most recent track for this object
                    if tracks:
                        latest_track = tracks[-1]
                        stable_objects.append({
                            'bbox': latest_track['bbox'],
                            'confidence': latest_track['confidence'],
                            'class_name': obj_name,
                            'class_id': 0  # We'll get this from current detections if needed
                        })
        
        return stable_objects

def detect_objects_yolov8(frame, model, stable_detector):
    """Detect objects using YOLOv8 with ultra-stability and accuracy"""
    current_time = time.time()
    
    # Only detect every 0.3 seconds for stability
    if current_time - stable_detector.last_detection_time < stable_detector.detection_interval:
        return stable_detector.get_stable_detections([])
    
    stable_detector.last_detection_time = current_time
    
    try:
        results = model(frame, conf=stable_detector.min_confidence, iou=0.45)
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    detections.append({
                        'bbox': (x1, y1, x2 - x1, y2 - y1),
                        'confidence': confidence,
                        'class_name': class_name,
                        'class_id': class_id
                    })
        
        # Get ultra-stable detections with accuracy improvements
        stable_detections = stable_detector.get_stable_detections(detections)
        return stable_detections
        
    except Exception as e:
        print(f"Error in YOLOv8 detection: {e}")
        return stable_detector.get_stable_detections([])

def draw_detections_yolov8(frame, detections):
    """Draw YOLOv8 detection results with ultra-stable labels"""
    for detection in detections:
        x, y, w, h = detection['bbox']
        confidence = detection['confidence']
        class_name = detection['class_name']
        
        # Use consistent colors for same objects
        color = (0, 255, 0)  # Green for stable detections
        
        # Draw thicker rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        
        # Create background for text
        text = f"{class_name}: {confidence:.2f}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        
        # Draw text background with padding
        cv2.rectangle(frame, (x, y - text_size[1] - 20),
                    (x + text_size[0] + 15, y), color, -1)
        
        # Draw text border for better visibility
        cv2.putText(frame, text, (x + 7, y - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
        cv2.putText(frame, text, (x + 7, y - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

def main():
    model = load_yolov8()
    if model is None:
        print("Failed to load YOLOv8 model. Exiting...")
        input("Press Enter to exit...")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        input("Press Enter to exit...")
        return

    # Initialize ultra-stable detector
    stable_detector = UltraStableDetector()

    print("Camera opened successfully!")
    print("Press 'q' to quit")
    print("YOLOv8 is detecting objects with ULTRA-ACCURATE & STABLE detection...")
    print("Better accuracy with maintained stability!")

    frame_count = 0
    last_print_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting...")
            break

        frame_count += 1
        
        # Detect objects with ultra-stability and accuracy
        detections = detect_objects_yolov8(frame, model, stable_detector)
        draw_detections_yolov8(frame, detections)
        
        # Print detections less frequently
        current_time = time.time()
        if detections and current_time - last_print_time > 3.0:  # Print every 3 seconds
            detected_objects = [det['class_name'] for det in detections]
            print(f"Ultra-Accurate & Stable Detections: {', '.join(detected_objects)}")
            last_print_time = current_time

        cv2.imshow('YOLOv8 Ultra-Accurate & Stable Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quit requested by user")
            break

    print("Cleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    print("Program finished successfully!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", str(e))
        import traceback
        traceback.print_exc()
    finally:
        input("Press Enter to exit...") 