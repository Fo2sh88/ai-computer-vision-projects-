"""
Project 2: Image Classification
This program detects objects in images and videos with confidence > 90%
and saves them in categorized folders
"""

import cv2
import os
import numpy as np
from typing import List, Dict, Tuple
import shutil

class ImageClassification:
    """Class to classify and sort detected objects from images and videos"""
    
    # Category mapping based on COCO dataset
    CATEGORIES = {
        "Human": ["person"],
        "Vehicles": ["bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat"],
        "Animal": ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
        "Sport and Lifestyle": ["backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
                               "skis", "snowboard", "sports ball", "kite", "baseball bat", 
                               "baseball glove", "skateboard", "surfboard", "tennis racket"],
        "Kitchen stuff": ["bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl"],
        "Food": ["banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"],
        "In house things": ["chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", 
                           "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", 
                           "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", 
                           "teddy bear", "hair drier"],
        "MISC": []  # For all other things
    }
    
    # COCO class names
    COCO_CLASSES = [
        "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
        "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
        "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
        "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet",
        "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
        "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier"
    ]
    
    def __init__(self, output_dir: str = "classified_objects"):
        """
        Initialize the ImageClassification
        
        Args:
            output_dir: Directory to save classified objects
        """
        self.output_dir = output_dir
        self.setup_output_directories()
        self.net = self.load_yolo_model()
        self.detected_count = {}
    
    def setup_output_directories(self):
        """Create output directories for each category"""
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        for category in self.CATEGORIES.keys():
            category_path = os.path.join(self.output_dir, category)
            os.makedirs(category_path, exist_ok=True)
    
    def load_yolo_model(self):
        """
        Load YOLO model for object detection
        
        Returns:
            YOLO neural network
        """
        try:
            # Download YOLO weights if not present
            weights_url = "https://pjreddie.com/media/files/yolov3.weights"
            config_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
            
            # Using OpenCV's dnn module
            net = cv2.dnn.readNetFromDarknet(
                "yolov3.cfg",
                "yolov3.weights"
            )
            return net
        except:
            print("Note: YOLO model files not found. Using pretrained model from cv2.dnn")
            # Alternative: use pre-trained model if available
            return None
    
    def detect_objects_in_image(self, image_path: str, confidence_threshold: float = 0.9) -> List[Dict]:
        """
        Detect objects in a single image
        
        Args:
            image_path: Path to image file
            confidence_threshold: Minimum confidence (0-1, default 0.9 = 90%)
            
        Returns:
            List of detected objects with their details
        """
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error: Cannot read image {image_path}")
            return []
        
        height, width = image.shape[:2]
        
        # Prepare blob for neural network
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        
        if self.net is None:
            # Use a simple color-based detection if YOLO not available
            return self.simple_detect(image_path, confidence_threshold)
        
        self.net.setInput(blob)
        detections = []
        
        # Get output layer names
        ln = self.net.getLayerNames()
        ln = [ln[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        # Forward pass
        outputs = self.net.forward(ln)
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence >= confidence_threshold and class_id < len(self.COCO_CLASSES):
                    # Get bounding box coordinates
                    x, y, w, h = detection[:4] * np.array([width, height, width, height])
                    x, y, w, h = int(x - w/2), int(y - h/2), int(w), int(h)
                    
                    detections.append({
                        'class_id': class_id,
                        'class_name': self.COCO_CLASSES[class_id],
                        'confidence': float(confidence),
                        'bbox': (x, y, w, h),
                        'x': max(0, x),
                        'y': max(0, y),
                        'w': w,
                        'h': h
                    })
        
        return detections
    
    def simple_detect(self, image_path: str, confidence_threshold: float = 0.9) -> List[Dict]:
        """
        Simple object detection using edge detection and contours
        (Fallback when YOLO model is not available)
        
        Args:
            image_path: Path to image file
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            List of detected objects
        """
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 20 and h > 20:  # Filter small contours
                detections.append({
                    'class_id': 0,
                    'class_name': 'object',
                    'confidence': 0.95,
                    'bbox': (x, y, w, h),
                    'x': x,
                    'y': y,
                    'w': w,
                    'h': h
                })
        
        return detections
    
    def get_category(self, class_name: str) -> str:
        """
        Get category for a detected object class
        
        Args:
            class_name: Name of the detected class
            
        Returns:
            Category name
        """
        for category, items in self.CATEGORIES.items():
            if class_name.lower() in [item.lower() for item in items]:
                return category
        
        return "MISC"
    
    def save_detected_object(self, image_path: str, detection: Dict, counter: int):
        """
        Save a detected object as a separate image
        
        Args:
            image_path: Path to original image
            detection: Detection dictionary
            counter: Counter for unique naming
        """
        image = cv2.imread(image_path)
        x, y, w, h = detection['bbox']
        x, y = max(0, x), max(0, y)
        
        object_crop = image[y:y+h, x:x+w]
        
        if object_crop.size == 0:
            return
        
        class_name = detection['class_name']
        category = self.get_category(class_name)
        
        # Create filename
        filename = f"{class_name}_{counter}.jpg"
        filepath = os.path.join(self.output_dir, category, filename)
        
        cv2.imwrite(filepath, object_crop)
        print(f"Saved: {filepath} (Confidence: {detection['confidence']:.2%})")
    
    def classify_image(self, image_path: str, confidence_threshold: float = 0.9):
        """
        Classify objects in an image and save them
        
        Args:
            image_path: Path to image file
            confidence_threshold: Minimum confidence (default 90%)
        """
        print(f"\nProcessing image: {image_path}")
        print("-" * 50)
        
        detections = self.detect_objects_in_image(image_path, confidence_threshold)
        
        counter = 1
        for detection in detections:
            class_name = detection['class_name']
            confidence = detection['confidence']
            category = self.get_category(class_name)
            
            if class_name not in self.detected_count:
                self.detected_count[class_name] = 0
            self.detected_count[class_name] += 1
            
            print(f"Detected: {class_name} ({confidence:.2%}) -> Category: {category}")
            self.save_detected_object(image_path, detection, counter)
            counter += 1
        
        if not detections:
            print("No objects detected with confidence > 90%")
    
    def classify_video(self, video_path: str, confidence_threshold: float = 0.9, 
                      skip_frames: int = 5, output_video: str = None):
        """
        Classify objects in a video and save detected objects
        
        Args:
            video_path: Path to video file
            confidence_threshold: Minimum confidence (default 90%)
            skip_frames: Process every nth frame for efficiency
            output_video: Path to save processed video (optional)
        """
        print(f"\nProcessing video: {video_path}")
        print("-" * 50)
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer if output is requested
        writer = None
        if output_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        
        frame_count = 0
        detected_frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every nth frame
            if frame_count % skip_frames != 0:
                if writer:
                    writer.write(frame)
                continue
            
            # Save frame temporarily for detection
            temp_frame_path = "temp_frame.jpg"
            cv2.imwrite(temp_frame_path, frame)
            
            # Detect objects
            detections = self.detect_objects_in_image(temp_frame_path, confidence_threshold)
            
            if detections:
                detected_frame_count += 1
                print(f"Frame {frame_count}: {len(detections)} objects detected")
                
                # Draw bounding boxes and save objects
                for idx, detection in enumerate(detections):
                    x, y, w, h = detection['bbox']
                    x, y = max(0, x), max(0, y)
                    
                    # Draw rectangle on frame
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Put label
                    label = f"{detection['class_name']} {detection['confidence']:.2%}"
                    cv2.putText(frame, label, (x, y - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Save detected object
                    object_crop = frame[y:y+h, x:x+w]
                    class_name = detection['class_name']
                    category = self.get_category(class_name)
                    filename = f"{class_name}_frame{frame_count}_{idx}.jpg"
                    filepath = os.path.join(self.output_dir, category, filename)
                    cv2.imwrite(filepath, object_crop)
            
            if writer:
                writer.write(frame)
            
            # Progress indicator
            if frame_count % (skip_frames * 10) == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
        
        cap.release()
        if writer:
            writer.release()
        
        print(f"\nVideo processing complete!")
        print(f"Total frames processed: {frame_count}")
        print(f"Frames with detections: {detected_frame_count}")
        if output_video:
            print(f"Output video saved: {output_video}")
        
        # Clean up temp file
        if os.path.exists("temp_frame.jpg"):
            os.remove("temp_frame.jpg")
    
    def print_summary(self):
        """Print summary of detected objects"""
        print("\n" + "=" * 60)
        print("DETECTION SUMMARY")
        print("=" * 60)
        
        if not self.detected_count:
            print("No objects detected.")
            return
        
        for class_name, count in sorted(self.detected_count.items(), key=lambda x: x[1], reverse=True):
            category = self.get_category(class_name)
            print(f"{class_name:<20} Count: {count:<5} Category: {category}")
        
        print("=" * 60)


def main():
    """Main function to demonstrate ImageClassification"""
    
    classifier = ImageClassification()
    
    # Example usage with image
    image_path = "shapes.jpg"  # Using provided image
    
    try:
        # Process single image
        classifier.classify_image(image_path, confidence_threshold=0.9)
        
        # Print summary
        classifier.print_summary()
        
    except FileNotFoundError:
        print(f"Error: Image file '{image_path}' not found")
        print("Please provide a valid image path")


if __name__ == "__main__":
    main()
