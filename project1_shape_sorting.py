"""
Project 1: Sorting objects by area
This program takes an image containing different shapes and:
1. Detects shapes in the image
2. Calculates the area of each shape
3. Generates a descending sorted list per area of shapes found with their names
"""

import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

class ShapeSorter:
    """Class to detect and sort shapes by area"""
    
    def __init__(self, image_path: str):
        """
        Initialize the ShapeSorter with an image
        
        Args:
            image_path: Path to the image file
        """
        self.image = cv2.imread(image_path)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.shapes = []
    
    def identify_shape(self, approx: np.ndarray) -> str:
        """
        Identify shape based on number of vertices
        
        Args:
            approx: Approximated contour
            
        Returns:
            Shape name (triangle, square, pentagon, hexagon, circle)
        """
        vertices = len(approx)
        
        if vertices == 3:
            return "triangle"
        elif vertices == 4:
            # Calculate aspect ratio to distinguish square from rectangle
            (x, y, w, h) = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.85 < aspect_ratio < 1.15:
                return "square"
            else:
                return "rectangle"
        elif vertices == 5:
            return "pentagon"
        elif vertices == 6:
            return "hexagon"
        elif vertices >= 7:
            return "circle"
        else:
            return "unknown"
    
    def calculate_area(self, contour: np.ndarray) -> float:
        """
        Calculate the area of a contour
        
        Args:
            contour: Contour points
            
        Returns:
            Area value
        """
        return cv2.contourArea(contour)
    
    def detect_shapes(self, min_area: float = 500.0) -> List[Dict]:
        """
        Detect all shapes in the image
        
        Args:
            min_area: Minimum area threshold to filter small shapes
            
        Returns:
            List of dictionaries containing shape info
        """
        # Threshold the image
        _, binary = cv2.threshold(self.gray, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        self.shapes = []
        
        for contour in contours:
            area = self.calculate_area(contour)
            
            # Filter by minimum area
            if area < min_area:
                continue
            
            # Approximate contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Identify shape
            shape_name = self.identify_shape(approx)
            
            self.shapes.append({
                'name': shape_name,
                'area': area,
                'contour': contour,
                'approx': approx
            })
        
        return self.shapes
    
    def sort_shapes_by_area(self, descending: bool = True) -> List[Tuple[int, str, float]]:
        """
        Sort shapes by area
        
        Args:
            descending: If True, sort in descending order (largest first)
            
        Returns:
            Sorted list of (rank, shape_name, area)
        """
        sorted_shapes = sorted(self.shapes, key=lambda x: x['area'], reverse=descending)
        
        result = []
        for i, shape in enumerate(sorted_shapes, 1):
            result.append((i, shape['name'], shape['area']))
        
        return result
    
    def visualize_results(self, output_path: str = None):
        """
        Visualize detected shapes with annotations
        
        Args:
            output_path: Path to save the visualization image
        """
        result_image = self.image.copy()
        
        sorted_shapes = self.sort_shapes_by_area()
        
        for rank, shape_info in enumerate(self.shapes, 1):
            contour = shape_info['contour']
            approx = shape_info['approx']
            
            # Draw contour
            cv2.drawContours(result_image, [contour], 0, (0, 255, 0), 2)
            
            # Find bounding box for text placement
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Put shape name and rank
                text = f"{shape_info['name']} {shape_info['area']:.0f}"
                cv2.putText(result_image, text, (cx - 30, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        if output_path:
            cv2.imwrite(output_path, result_image)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title('Detected Shapes')
        plt.tight_layout()
        plt.show()
    
    def get_sorted_list(self) -> List[Tuple[int, str, float]]:
        """
        Get the final sorted list of shapes by area
        
        Returns:
            List of (rank, shape_name, area)
        """
        return self.sort_shapes_by_area(descending=True)
    
    def print_results(self):
        """Print the sorted shapes list in a formatted way"""
        print("=" * 60)
        print("SHAPE SORTING RESULTS (BY AREA - DESCENDING)")
        print("=" * 60)
        print(f"{'Rank':<6}{'Shape Name':<15}{'Area (pixels)':<20}")
        print("-" * 60)
        
        sorted_shapes = self.get_sorted_list()
        
        for rank, name, area in sorted_shapes:
            print(f"{rank:<6}{name:<15}{area:<20.2f}")
        
        print("=" * 60)


def main():
    """Main function to demonstrate the ShapeSorter"""
    
    # Example usage with the shapes.jpg provided
    image_path = "shapes.jpg"  # Using the provided image
    
    try:
        sorter = ShapeSorter(image_path)
        
        # Detect shapes
        sorter.detect_shapes(min_area=500)
        
        # Print results
        sorter.print_results()
        
        # Visualize results
        sorter.visualize_results()
        
    except FileNotFoundError:
        print(f"Error: Image file '{image_path}' not found")
        print("Please ensure the image file is in the same directory as this script")


if __name__ == "__main__":
    main()
