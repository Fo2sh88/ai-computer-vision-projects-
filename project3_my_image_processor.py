"""
Project 3: MyImageProcessor
A comprehensive image processing class using NumPy, OpenCV, and Matplotlib
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Tuple
import os

class MyImageProcessor:
    """
    Image processor class with various image manipulation methods
    """
    
    def __init__(self, image_path: str):
        """
        Constructor that reads an image file
        
        Args:
            image_path: Path to the image file (str)
            
        Raises:
            FileNotFoundError: If image file doesn't exist
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        self.image_path = image_path
        self.image_bgr = cv2.imread(image_path)
        
        if self.image_bgr is None:
            raise ValueError(f"Failed to read image: {image_path}")
        
        print(f"Image loaded successfully: {image_path}")
        print(f"Image shape (H, W, C): {self.image_bgr.shape}")
    
    def bgr_2_rgb_convertor(self) -> np.ndarray:
        """
        Convert BGR image to RGB and display it
        
        Returns:
            np.ndarray: RGB image as numpy array
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2RGB)
        
        # Display the image
        plt.figure(figsize=(10, 8))
        plt.imshow(rgb_image)
        plt.title("RGB Image")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        print(f"RGB image shape: {rgb_image.shape}")
        return rgb_image
    
    def bgr_2_gray_scale_convertor(self) -> np.ndarray:
        """
        Convert BGR image to grayscale and display it
        
        Returns:
            np.ndarray: Grayscale image as numpy array
        """
        # Convert BGR to Grayscale
        gray_image = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2GRAY)
        
        # Display the image
        plt.figure(figsize=(10, 8))
        plt.imshow(gray_image, cmap='gray')
        plt.title("Grayscale Image")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        print(f"Grayscale image shape: {gray_image.shape}")
        return gray_image
    
    def _50_percent_resizer(self) -> np.ndarray:
        """
        Resize image to 50% of original width and height, display in RGB
        
        Returns:
            np.ndarray: Resized RGB image as numpy array
        """
        # Calculate new dimensions
        height, width = self.image_bgr.shape[:2]
        new_width = width // 2
        new_height = height // 2
        
        # Resize the image
        resized_image = cv2.resize(self.image_bgr, (new_width, new_height))
        
        # Convert to RGB for display
        resized_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        
        # Display the image
        plt.figure(figsize=(10, 8))
        plt.imshow(resized_rgb)
        plt.title(f"50% Resized Image ({width}x{height} → {new_width}x{new_height})")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        print(f"Original size: {width}x{height}")
        print(f"Resized to 50%: {new_width}x{new_height}")
        print(f"Resized image shape: {resized_rgb.shape}")
        
        return resized_rgb
    
    def image_writer(self, output_image_path_and_name: str) -> bool:
        """
        Save the image in RGB format
        
        Args:
            output_image_path_and_name: Output path and filename (str)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2RGB)
            
            # Convert RGB back to BGR for saving (since cv2.imwrite uses BGR)
            bgr_for_save = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            
            # Create directory if it doesn't exist
            output_dir = os.path.dirname(output_image_path_and_name)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # Save the image
            success = cv2.imwrite(output_image_path_and_name, bgr_for_save)
            
            if success:
                print(f"Image saved successfully: {output_image_path_and_name}")
            else:
                print(f"Failed to save image: {output_image_path_and_name}")
            
            return success
        
        except Exception as e:
            print(f"Error saving image: {e}")
            return False
    
    def frame_it(self, output_image_with_frame_path: str) -> np.ndarray:
        """
        Draw a RED frame (rectangle) around the image with 20px thickness
        Save and return as numpy array in RGB
        
        Args:
            output_image_with_frame_path: Output path and filename (str)
            
        Returns:
            np.ndarray: RGB image with frame as numpy array
        """
        # Create a copy of the image
        framed_image = self.image_bgr.copy()
        
        # Get image dimensions
        height, width = framed_image.shape[:2]
        
        # Draw RED rectangle frame with 20px thickness
        # Red in BGR is (0, 0, 255)
        cv2.rectangle(framed_image, (0, 0), (width - 1, height - 1), (0, 0, 255), 20)
        
        # Convert to RGB for display and return
        framed_rgb = cv2.cvtColor(framed_image, cv2.COLOR_BGR2RGB)
        
        # Display the image
        plt.figure(figsize=(10, 8))
        plt.imshow(framed_rgb)
        plt.title("Image with RED Frame (20px)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        # Save the image in RGB format
        bgr_for_save = cv2.cvtColor(framed_rgb, cv2.COLOR_RGB2BGR)
        
        # Create directory if needed
        output_dir = os.path.dirname(output_image_with_frame_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        cv2.imwrite(output_image_with_frame_path, bgr_for_save)
        print(f"Framed image saved: {output_image_with_frame_path}")
        
        return framed_rgb
    
    def find_center(self, output_image_with_center: str) -> np.ndarray:
        """
        Draw a BLUE point at the center of the image with text "image center"
        Save and return as numpy array in RGB
        
        Args:
            output_image_with_center: Output path and filename (str)
            
        Returns:
            np.ndarray: RGB image with center point as numpy array
        """
        # Create a copy of the image
        center_image = self.image_bgr.copy()
        
        # Get image dimensions
        height, width = center_image.shape[:2]
        
        # Calculate center coordinates
        center_x = width // 2
        center_y = height // 2
        
        # Draw BLUE point (circle) at center
        # Blue in BGR is (255, 0, 0)
        radius = 10
        thickness = -1  # Filled circle
        cv2.circle(center_image, (center_x, center_y), radius, (255, 0, 0), thickness)
        
        # Put text "image center" with good font size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        font_color = (255, 0, 0)  # Blue in BGR
        font_thickness = 2
        
        text = "image center"
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = center_x - text_size[0] // 2
        text_y = center_y - radius - 20
        
        cv2.putText(center_image, text, (text_x, text_y), font, 
                   font_scale, font_color, font_thickness)
        
        # Convert to RGB for display and return
        center_rgb = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
        
        # Display the image
        plt.figure(figsize=(10, 8))
        plt.imshow(center_rgb)
        plt.title(f"Image with Center Point at ({center_x}, {center_y})")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        # Save the image in RGB format
        bgr_for_save = cv2.cvtColor(center_rgb, cv2.COLOR_RGB2BGR)
        
        # Create directory if needed
        output_dir = os.path.dirname(output_image_with_center)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        cv2.imwrite(output_image_with_center, bgr_for_save)
        print(f"Center point image saved: {output_image_with_center}")
        
        return center_rgb
    
    def detect_faces(self) -> Tuple[np.ndarray, int]:
        """
        Detect faces using CascadeClassifier
        Draw RED rectangles around detected faces
        
        Returns:
            Tuple[np.ndarray, int]: (image with face rectangles as numpy array, face count)
        """
        # Load the cascade classifier
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Create a copy of the original image
        face_detected_image = self.image_bgr.copy()
        
        # Draw RED rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(face_detected_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            # Add face number label
            label = f"Face {len(faces)}"
            cv2.putText(face_detected_image, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        # Convert to RGB
        face_rgb = cv2.cvtColor(face_detected_image, cv2.COLOR_BGR2RGB)
        
        # Display the result
        plt.figure(figsize=(12, 8))
        plt.imshow(face_rgb)
        plt.title(f"Face Detection - {len(faces)} face(s) detected")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        faces_counter = len(faces)
        print(f"Faces detected: {faces_counter}")
        
        return face_rgb, faces_counter


def main():
    """Main function to demonstrate MyImageProcessor"""
    
    try:
        # Initialize with the provided image
        image_path = "shapes.jpg"
        processor = MyImageProcessor(image_path)
        
        print("\n" + "=" * 60)
        print("MyImageProcessor Demonstration")
        print("=" * 60)
        
        # Test each method
        print("\n1. Converting BGR to RGB...")
        rgb_image = processor.bgr_2_rgb_convertor()
        
        print("\n2. Converting to Grayscale...")
        gray_image = processor.bgr_2_gray_scale_convertor()
        
        print("\n3. Resizing to 50%...")
        resized_image = processor._50_percent_resizer()
        
        print("\n4. Saving image...")
        processor.image_writer("output_rgb.jpg")
        
        print("\n5. Adding frame...")
        framed_image = processor.frame_it("output_framed.jpg")
        
        print("\n6. Finding center...")
        center_image = processor.find_center("output_center.jpg")
        
        print("\n7. Detecting faces...")
        face_image, face_count = processor.detect_faces()
        
        print("\n" + "=" * 60)
        print("Processing Complete!")
        print("=" * 60)
        print(f"\nOutput files created:")
        print(f"  - output_rgb.jpg")
        print(f"  - output_framed.jpg")
        print(f"  - output_center.jpg")
        print(f"  - Faces detected: {face_count}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the image file exists in the current directory")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
