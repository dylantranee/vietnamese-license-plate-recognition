"""
Vietnamese License Plate Recognition - Image Processor

This script processes single images to detect and recognize Vietnamese license plates.
It uses the same recognition pipeline as the main video processor but optimized for
static images with enhanced filtering to reduce false positives.

Features:
- Smart filtering based on aspect ratio and size
- Multi-plate detection support
- Command-line interface for batch processing

Usage:
    python image_processor.py --image path/to/image.jpg --output result.jpg
    
Example:
    python image_processor.py --image tests/images/motorcycles/Hung_0242.png
"""

import cv2
import argparse
import sys
from main import model, recognize_plate

def process_image(image_path, output_path="output_image.jpg"):
    """
    Detects and recognizes license plates in a single image.
    
    Args:
        image_path: Path to input image file
        output_path: Path to save annotated output image
        
    Process:
        1. Load image
        2. Run YOLO detection
        3. Filter detections by confidence, aspect ratio, and size
        4. Recognize plate text for each valid detection
        5. Draw annotations and save output
    """
    # Load input image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image at '{image_path}'. Please check the path.")
        return

    print(f"Processing {image_path}...")

    # Run YOLO inference
    results = model(frame, verbose=False)
    
    # Detection parameters
    CONF_THRESH = 0.3  # Minimum confidence threshold
    detections = 0

    for r in results:
        boxes = r.boxes
        
        # Smart filtering: Remove false positives based on multiple criteria
        valid_boxes = []
        for box in boxes:
            conf = float(box.conf.item())
            
            # Skip low-confidence detections
            if conf < CONF_THRESH:
                continue
            
            # Extract bounding box dimensions
            x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = width / height if height > 0 else 0
            area = width * height
            
            # Filter criteria for valid license plates:
            # 1. Confidence >= 0.3 (already checked above)
            # 2. Plate-like aspect ratio (0.8 - 5.0) - handles various angles
            # 3. Minimum area >= 500 pixels - filters out tiny false detections
            if (0.8 <= aspect_ratio <= 5.0 and area >= 500):
                valid_boxes.append(box)
        
        # Process only valid detections
        for box in valid_boxes:
            detections += 1
            x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
            
            # Crop the plate region
            plate_crop = frame[y1:y2, x1:x2]

            # Recognize plate text using shared recognition pipeline
            text = recognize_plate(plate_crop)

            # Draw green bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Draw recognized text label if plate was successfully recognized
            if text:
                # Calculate text size to ensure it fits within image bounds
                (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                
                # Position text above the bounding box, ensuring it stays within image bounds
                label_y = max(text_height + 10, y1 - 10)  # At least text_height from top
                label_x = min(x1, frame.shape[1] - text_width - 10)  # Don't exceed right edge
                label_x = max(10, label_x)  # Don't exceed left edge
                
                # Draw text with shadow for readability
                cv2.putText(frame, text, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
                cv2.putText(frame, text, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                
                print(f" [Matched] Plate: {text}")
            else:
                print(f" [Ignored] Detected box but no valid text.")

    if detections == 0:
        print("No license plates detected.")
    
    # Save annotated output image
    cv2.imwrite(output_path, frame)
    print(f"Annotated image saved to {output_path}")

if __name__ == "__main__":
    # Setup command-line argument parser
    parser = argparse.ArgumentParser(description="License Plate Recognition on Single Image")
    parser.add_argument("--image", required=True, type=str, help="Path to input image file")
    parser.add_argument("--output", default="output_image.jpg", type=str, help="Path to save annotated output image")
    
    args = parser.parse_args()
    process_image(args.image, args.output)
