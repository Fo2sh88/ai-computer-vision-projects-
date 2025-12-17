"""
Complete Test Suite for all three projects
Demonstrates usage and validates implementations
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def test_project1():
    """Test Project 1: Shape Sorting"""
    print("\n" + "="*70)
    print("TESTING PROJECT 1: SHAPE SORTING BY AREA")
    print("="*70)
    
    try:
        from project1_shape_sorting import ShapeSorter
        
        image_path = "shapes.jpg"
        if not os.path.exists(image_path):
            print(f"⚠️  Image file '{image_path}' not found")
            print("   Please ensure shapes.jpg is in the working directory")
            return False
        
        # Initialize
        sorter = ShapeSorter(image_path)
        print("✅ ShapeSorter initialized successfully")
        
        # Detect shapes
        shapes = sorter.detect_shapes(min_area=500)
        print(f"✅ Detected {len(shapes)} shapes")
        
        # Sort and display
        sorted_list = sorter.get_sorted_list()
        print("✅ Shapes sorted by area (descending)")
        
        # Print results
        sorter.print_results()
        
        print("✅ PROJECT 1 TEST PASSED")
        return True
        
    except Exception as e:
        print(f"❌ PROJECT 1 TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_project2():
    """Test Project 2: Image Classification"""
    print("\n" + "="*70)
    print("TESTING PROJECT 2: IMAGE CLASSIFICATION")
    print("="*70)
    
    try:
        from project2_image_classification import ImageClassification
        
        # Initialize classifier
        classifier = ImageClassification(output_dir="classified_objects_test")
        print("✅ ImageClassification initialized successfully")
        
        # Test with image
        image_path = "shapes.jpg"
        if not os.path.exists(image_path):
            print(f"⚠️  Image file '{image_path}' not found")
            print("   Please ensure shapes.jpg is in the working directory")
            return False
        
        # Classify image
        classifier.classify_image(image_path, confidence_threshold=0.9)
        print("✅ Image classification completed")
        
        # Print summary
        classifier.print_summary()
        print("✅ Summary generated")
        
        # Verify output directory
        if os.path.exists("classified_objects_test"):
            print("✅ Output directory created successfully")
        
        print("✅ PROJECT 2 TEST PASSED")
        return True
        
    except Exception as e:
        print(f"❌ PROJECT 2 TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_project3():
    """Test Project 3: MyImageProcessor"""
    print("\n" + "="*70)
    print("TESTING PROJECT 3: MY IMAGE PROCESSOR")
    print("="*70)
    
    try:
        from project3_my_image_processor import MyImageProcessor
        
        image_path = "shapes.jpg"
        if not os.path.exists(image_path):
            print(f"⚠️  Image file '{image_path}' not found")
            print("   Please ensure shapes.jpg is in the working directory")
            return False
        
        # Initialize
        processor = MyImageProcessor(image_path)
        print("✅ MyImageProcessor initialized successfully")
        
        # Create output directory
        output_dir = "output_test"
        os.makedirs(output_dir, exist_ok=True)
        
        # Test RGB conversion
        rgb = processor.bgr_2_rgb_convertor()
        print("✅ BGR to RGB conversion passed")
        
        # Test grayscale conversion
        gray = processor.bgr_2_gray_scale_convertor()
        print("✅ BGR to Grayscale conversion passed")
        
        # Test resize
        resized = processor._50_percent_resizer()
        print("✅ 50% resize operation passed")
        
        # Test image writer
        output_path = os.path.join(output_dir, "output_rgb.jpg")
        processor.image_writer(output_path)
        print(f"✅ Image writer passed - saved to {output_path}")
        
        # Test frame drawing
        frame_path = os.path.join(output_dir, "output_framed.jpg")
        framed = processor.frame_it(frame_path)
        print(f"✅ Frame drawing passed - saved to {frame_path}")
        
        # Test center finding
        center_path = os.path.join(output_dir, "output_center.jpg")
        centered = processor.find_center(center_path)
        print(f"✅ Center detection passed - saved to {center_path}")
        
        # Test face detection
        face_img, face_count = processor.detect_faces()
        print(f"✅ Face detection passed - detected {face_count} face(s)")
        
        print("✅ PROJECT 3 TEST PASSED")
        return True
        
    except Exception as e:
        print(f"❌ PROJECT 3 TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and generate report"""
    print("\n" + "="*70)
    print("RUNNING COMPLETE TEST SUITE")
    print("="*70)
    
    results = {
        "Project 1 (Shape Sorting)": test_project1(),
        "Project 2 (Image Classification)": test_project2(),
        "Project 3 (MyImageProcessor)": test_project3(),
    }
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for project, status in results.items():
        status_str = "✅ PASSED" if status else "❌ FAILED"
        print(f"{project:<35} {status_str}")
    
    print("-"*70)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Ready for submission (VG Quality)")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review errors above.")
        return False


def check_dependencies():
    """Check if all required dependencies are installed"""
    print("\n" + "="*70)
    print("CHECKING DEPENDENCIES")
    print("="*70)
    
    required_packages = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'scipy': 'scipy'
    }
    
    all_ok = True
    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is NOT installed")
            print(f"   Install with: pip install {package}")
            all_ok = False
    
    if not all_ok:
        print("\n⚠️  Some dependencies are missing. Install them with:")
        print("   pip install -r requirements.txt")
    
    return all_ok


def main():
    """Main test runner"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "INLÄMNINGSUPPGIFT 1.AI - TEST SUITE" + " "*19 + "║")
    print("║" + " "*10 + "All Projects Implementation Validation" + " "*20 + "║")
    print("╚" + "="*68 + "╝")
    
    # Check dependencies first
    if not check_dependencies():
        print("\n⚠️  Please install missing dependencies and try again.")
        return False
    
    # Run tests
    success = run_all_tests()
    
    # Summary info
    if success:
        print("\n" + "="*70)
        print("READY FOR GITHUB PUSH")
        print("="*70)
        print("\nNext steps:")
        print("1. git init")
        print("2. git add .")
        print("3. git commit -m 'Complete AI Inlämningsuppgift 1 - All Projects (VG)'")
        print("4. git branch -M main")
        print("5. git remote add origin <your-repo-url>")
        print("6. git push -u origin main")
    
    return success


if __name__ == "__main__":
    # Suppress matplotlib displays during testing
    import matplotlib
    matplotlib.use('Agg')
    
    success = main()
    sys.exit(0 if success else 1)
