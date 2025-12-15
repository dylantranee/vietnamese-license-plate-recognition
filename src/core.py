"""
Core Logic for Vietnamese License Plate Recognition

This module contains the shared business logic for detecting and acting on license plates,
independent of the input source (image vs video).
"""

import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import re

# Import province code mapping utility
try:
    from utils.province_codes import get_province_name
except ImportError:
    def get_province_name(code): 
        return ""

# ============================================================================
# 1. MODEL INITIALIZATION
# ============================================================================

# Load fine-tuned YOLOv8 model for license plate detection
# Using a function to allow lazy loading or external control if needed, 
# but keeping global for simplicity in this script.
print("Loading models...")
model = YOLO("models/license_plate_best.pt")

# Initialize EasyOCR reader with English and Vietnamese language support
# GPU acceleration enabled for faster processing
reader = easyocr.Reader(['en', 'vi'], gpu=True)
print("Models loaded successfully.")

# ============================================================================
# 2. PLATE FORMAT VALIDATION PATTERNS
# ============================================================================

# Regex pattern for motorcycle plates: DD-LN-XXX.XX
# DD: Province code (11-99), L: Letter, N: Digit (1-9), XXX.XX: Registration number
MOTO_PATTERN = re.compile(r"^(1[1-9]|[2-9]\d)-[A-Z][1-9]-(0\d{2}\.[0-9]{2}|[1-9]\d{2}\.[0-9]{2})$")

# Regex pattern for car plates: DD-L-XXX.XX
# DD: Province code (11-99), L: Letter, XXX.XX: Registration number
CAR_PATTERN = re.compile(r"^(1[1-9]|[2-9]\d)-[A-Z]-(0\d{2}\.[0-9]{2}|[1-9]\d{2}\.[0-9]{2})$")

# ============================================================================
# 3. IMAGE PREPROCESSING FUNCTIONS
# ============================================================================

def order_points(pts):
    """
    Orders 4 corner points in clockwise order: top-left, top-right, bottom-right, bottom-left.
    Used for perspective transformation.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left has smallest sum
    rect[2] = pts[np.argmax(s)]  # Bottom-right has largest sum

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right has smallest difference
    rect[3] = pts[np.argmax(diff)]  # Bottom-left has largest difference
    return rect

def deskew_plate(image):
    """
    Corrects perspective distortion using 4-point contour detection.
    Attempts to find the plate boundary and apply perspective transform.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detect edges using Canny edge detection
    edged = cv2.Canny(blurred, 50, 200, 255)
    
    # Find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    display_cnt = None
    image_area = image.shape[0] * image.shape[1]

    # Find the largest 4-sided contour (likely the plate boundary)
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # Check if contour has 4 sides and covers significant area (>50%)
        if len(approx) == 4:
            if cv2.contourArea(approx) > 0.5 * image_area:
                display_cnt = approx
                break

    # If no suitable contour found, return original image
    if display_cnt is None:
        # print("DEBUG: No 4-point contour found for perspective transform.")
        return image

    # Order the corner points
    pts = display_cnt.reshape(4, 2)
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Calculate dimensions of the warped image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Define destination points for perspective transform
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # Apply perspective transformation
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped

def preprocess_plate(plate_crop):
    """
    Applies comprehensive preprocessing to enhance OCR accuracy:
    1. Upscaling (5x) using Lanczos interpolation
    2. Border masking to remove plate frame artifacts
    3. Noise filtering with bilateral filter
    4. Contrast enhancement with CLAHE
    5. Otsu thresholding for binarization
    6. Contour-based noise removal
    7. Dynamic morphological operations based on character density
    
    Returns preprocessed binary image optimized for OCR.
    """
    if plate_crop is None or plate_crop.size == 0:
        return None
    
    # Step 1: Upscale image 5x for better detail recognition
    plate_resized = cv2.resize(plate_crop, None, fx=5, fy=5, interpolation=cv2.INTER_LANCZOS4)
    
    # Step 2: Mask border pixels to remove plate frame and screw artifacts
    # This prevents OCR from detecting phantom characters from the frame
    if len(plate_resized.shape) == 3:
        plate_resized[0:5, :] = (0, 0, 0)  # Top border
        plate_resized[-5:, :] = (0, 0, 0)  # Bottom border
        plate_resized[:, 0:5] = (0, 0, 0)  # Left border
        plate_resized[:, -5:] = (0, 0, 0)  # Right border
    else:
        plate_resized[0:5, :] = 0
        plate_resized[-5:, :] = 0
        plate_resized[:, 0:5] = 0
        plate_resized[:, -5:] = 0
    
    # Step 3: Deskewing disabled (causes issues with some plates)
    plate_deskewed = plate_resized

    # Gamma correction helper function (currently disabled)
    def adjust_gamma(image, gamma=1.0):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    # Apply gamma correction (disabled with gamma=1.0)
    plate_deskewed = adjust_gamma(plate_deskewed, gamma=1.0)
    
    # Step 4: Add white padding to help OCR detect edge characters
    plate_deskewed = cv2.copyMakeBorder(plate_deskewed, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    
    # Step 5: Convert to grayscale
    if len(plate_deskewed.shape) == 3:
        gray = cv2.cvtColor(plate_deskewed, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_deskewed
    
    # Step 6: Apply bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(gray, 11, 17, 17) 
    
    # Step 7: Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    plate_enhanced = clahe.apply(filtered) 
    
    # Step 8: Binarize image using Otsu's thresholding
    _, plate_binary = cv2.threshold(plate_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Step 9: Ensure white text on black background (invert if necessary)
    num_white = cv2.countNonZero(plate_binary)
    num_total = plate_binary.size
    if num_white > 0.5 * num_total:
         plate_binary = cv2.bitwise_not(plate_binary)
    
    # Step 10: Contour-based noise removal
    # Find all contours in the binary image
    cnts, _ = cv2.findContours(plate_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_area = plate_binary.shape[0] * plate_binary.shape[1]
    
    # Adaptive filtering: try strict threshold first, then relax if needed
    candidates = []
    area_thresholds = [1500, 500]  # Try strict (1500) first, then relaxed (500)
    
    for min_area in area_thresholds:
        candidates = []
        
        # Collect candidate contours (likely characters)
        for i, c in enumerate(cnts):
            (x, y, w, h) = cv2.boundingRect(c)
            aspect_ratio = w / float(h)
            area = cv2.contourArea(c)
            
            # Skip obvious noise
            is_noise = False
            if area < 30: is_noise = True  # Too small (specks)
            elif aspect_ratio < 0.35 and area < 600: is_noise = True  # Too thin (lines)
            elif aspect_ratio > 15: is_noise = True  # Too wide (horizontal lines)
                
            if not is_noise:
                 # Collect valid candidates (exclude massive borders and tiny noise)
                 if min_area < area < 0.5 * image_area:
                     candidates.append((c, h))
        
        # If we found enough candidates (>=4), stop trying lower thresholds
        if len(candidates) >= 4:
            break

    # Remove noise contours from the binary image
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(c)
        
        # Erase noise based on size and aspect ratio
        if area < 30: cv2.drawContours(plate_binary, [c], -1, 0, -1)
        elif aspect_ratio < 0.35 and area < 600: cv2.drawContours(plate_binary, [c], -1, 0, -1)
        elif aspect_ratio > 15: cv2.drawContours(plate_binary, [c], -1, 0, -1)

    # Step 11: Dynamic height-based filtering
    # Remove contours that are significantly shorter than the median height
    median_h = 0
    avg_density = 0
    
    if candidates:
        heights = [h for (_, h) in candidates]
        median_h = np.median(heights)
        
        # Remove outliers (contours much shorter than median)
        filtered_count = 0
        for (c, h) in candidates:
            (x, y, w_c, h_c) = cv2.boundingRect(c)
            ar_c = w_c / float(h_c)
            
            # Erase if height < 85% of median AND aspect ratio suggests it's a ghost line
            if h < 0.85 * median_h:
                if ar_c < 0.35:  # Thin ghost lines
                    cv2.drawContours(plate_binary, [c], -1, 0, -1)
                    filtered_count += 1
        
        # Calculate average pixel density of characters
        total_density = 0
        density_count = 0
        for (c, h) in candidates:
             (x, y, w, h) = cv2.boundingRect(c)
             roi = plate_binary[y:y+h, x:x+w]
             density = cv2.countNonZero(roi) / (w * h)
             total_density += density
             density_count += 1
        
        if density_count > 0:
             avg_density = total_density / density_count
        
    # Step 12: Dynamic morphological operations based on character size and density
    # Small characters (<130px): tend to be bloated from upscaling
    # Large characters (>=130px): tend to be thin/broken
    
    if 0 < median_h < 130:
        # Small text processing
        if avg_density >= 0.45 and median_h >= 80:
            # High density: characters are bloated, apply erosion to thin them
            kernel_erode = np.ones((2, 2), np.uint8)
            plate_binary = cv2.erode(plate_binary, kernel_erode, iterations=1)
        else:
            # Low density: characters are thin/broken, apply dilation to connect them
            if avg_density < 0.35:
                kernel_dilate = np.ones((3, 3), np.uint8)  # Stronger dilation for very thin chars
            else:
                kernel_dilate = np.ones((2, 2), np.uint8)  # Standard dilation
                
            plate_binary = cv2.dilate(plate_binary, kernel_dilate, iterations=1)
    else:
        # Large text processing (same density-based logic)
        if avg_density >= 0.45:
            kernel_erode = np.ones((2, 2), np.uint8)
            plate_binary = cv2.erode(plate_binary, kernel_erode, iterations=1)
        else:
            kernel_dilate = np.ones((2, 2), np.uint8)
            plate_binary = cv2.dilate(plate_binary, kernel_dilate, iterations=1)

    # Step 13: Invert to black text on white background (standard for OCR)
    plate_final = cv2.bitwise_not(plate_binary)
    
    return plate_final

# ============================================================================
# 4. CHARACTER CORRECTION AND VALIDATION
# ============================================================================

def correct_plate_format(ocr_text):
    """
    Cleans and validates OCR output to match Vietnamese plate formats.
    
    Process:
    1. Remove decorative text (VIETNAM, VN, etc.)
    2. Apply positional character correction (e.g., O→0 in province code)
    3. Format with dashes and dots
    4. Validate against regex patterns
    
    Returns formatted plate string (e.g., '51-C5-123.45') or empty string if invalid.
    """
    # Character confusion mappings for OCR error correction
    mapping_num_to_alpha = {"0": "O", "1": "A", "5": "S", "8": "B", "4": "A", "6": "G"}
    mapping_alpha_to_num = {"O": "0", "I": "1", "Z": "2", "S": "5", "B": "8", "U": "0"}
    mapping_num_to_n = {"1": "N"}  # Specific for motorcycle N-series

    # Step 1: Remove decorative text (common on Vietnamese plates)
    ocr_text_upper = ocr_text.upper()
    decorations = ["VIETNAM", "VIE", "VN"]
    for decoration in decorations:
        ocr_text_upper = ocr_text_upper.replace(decoration, "")
    
    # Step 2: Strip all punctuation and spaces
    ocr_text_clean = ocr_text_upper.replace(" ", "").replace(".", "").replace("-", "")
    length = len(ocr_text_clean)

    # Step 3: Validate length (7-10 characters)
    if length not in [7, 8, 9, 10]:
        return ""

    # Step 4: Pre-scan to detect 2-letter series format (positions 2 and 3 both letter candidates)
    # This helps us make smarter conversion decisions
    has_two_letter_series = False
    if length >= 4:
        pos2_is_letter_candidate = (ocr_text_clean[2].isalpha() or 
                                     (ocr_text_clean[2].isdigit() and ocr_text_clean[2] in mapping_num_to_alpha))
        pos3_is_letter_candidate = (ocr_text_clean[3].isalpha() or 
                                     (ocr_text_clean[3].isdigit() and ocr_text_clean[3] in mapping_num_to_alpha))
        
        # If both positions 2 and 3 look like letters, it's likely a 2-letter series (e.g., "29A7" → "29AA")
        # But we need to ensure position 3 is NOT a digit that should stay as digit (motorcycle LN format)
        if pos2_is_letter_candidate and pos3_is_letter_candidate:
            # Check if position 3 is actually a letter or a number that looks like a letter
            # We exclude 1,2,3,5,6,7,8,9 as definite digits. '0'->'O', '4'->'A'.
            if ocr_text_clean[3].isalpha() or (ocr_text_clean[3].isdigit() and ocr_text_clean[3] not in ['1', '2', '3', '5', '6', '7', '8', '9']):
                has_two_letter_series = True
    
    corrected = []
    
    # Step 5: Apply positional character correction
    for i, ch in enumerate(ocr_text_clean):
        # Positions 0-1: Province code (must be digits)
        if i == 0 or i == 1:
            if ch.isalpha() and ch in mapping_alpha_to_num:
                corrected.append(mapping_alpha_to_num[ch])
            elif ch.isdigit():
                corrected.append(ch)
            else:
                return ""  # Invalid character

        # Position 2: Serial letter (must be letter)
        elif i == 2:
            if ch.isdigit() and ch in mapping_num_to_alpha:
                # Special case: '1' → 'N' for motorcycle N-series (only if NOT 2-letter series)
                if ch == '1' and i + 1 < len(ocr_text_clean) and ocr_text_clean[i + 1].isdigit() and not has_two_letter_series:
                    corrected.append('N')
                else:
                    corrected.append(mapping_num_to_alpha[ch])
            elif ch.isalpha():
                corrected.append(ch)
            else:
                return ""
        
        # Position 3: Could be second letter (LL series), digit (LN series), or start of reg number
        elif i == 3:
            # If we detected 2-letter series, convert to letter
            if has_two_letter_series:
                if ch.isalpha():
                    corrected.append(ch)
                elif ch.isdigit() and ch in mapping_num_to_alpha:
                    corrected.append(mapping_num_to_alpha[ch])
                elif ch in ['I', 'L']:
                    corrected.append('I')  # Keep as letter for 2-letter series
                else:
                    return ""
            else:
                # Standard logic: could be digit or letter
                if ch.isalpha():
                    corrected.append(ch)
                elif ch.isdigit():
                    corrected.append(ch)
                elif ch in ['I', 'L']:
                    corrected.append('1')
                elif ch in mapping_alpha_to_num:
                    corrected.append(mapping_alpha_to_num[ch])
                else:
                    return ""
        
        # Positions 4+: Registration number (must be digits)
        else:
            if ch in ['I', 'L']:
                corrected.append('1')
            elif ch.isalpha() and ch in mapping_alpha_to_num:
                corrected.append(mapping_alpha_to_num[ch])
            elif ch.isdigit():
                corrected.append(ch)
            else:
                return ""

    candidate_raw = "".join(corrected)
    
    # Step 6: Contextual fix for 0↔9 confusion
    # Registration numbers starting with 02-08 are unlikely (should be 92-98)
    if len(candidate_raw) >= 5:
        reg_start = candidate_raw[3:5]
        if reg_start in ['02', '03', '04', '05', '06', '07', '08']:
            candidate_raw = candidate_raw[:3] + '9' + candidate_raw[4:]
    
    # Step 7: Remove phantom '1' artifact (common border detection error)
    prov_code = candidate_raw[0:2]
    if len(candidate_raw) == 9 and candidate_raw[3] == '1':
         temp_raw = candidate_raw[:3] + candidate_raw[4:]
         if len(temp_raw) == 8:
             t_serial = temp_raw[2]
             t_reg = temp_raw[3:8]
             t_str = f"{prov_code}-{t_serial}-{t_reg[:-2]}.{t_reg[-2:]}"
             if CAR_PATTERN.match(t_str):
                 candidate_raw = temp_raw
    
    # Step 8: Try different format interpretations based on length
    
    # Format 1: Special car with 2-letter series (DD-LL-XXX.XX)
    if len(candidate_raw) == 9 and candidate_raw[2].isalpha() and candidate_raw[3].isalpha():
        serial_letters = candidate_raw[2:4]
        reg_number_raw = candidate_raw[4:9]
        
        top_line = f"{prov_code}-{serial_letters}"
        reg_fmt = f"{reg_number_raw[:-2]}.{reg_number_raw[-2:]}"
        final_string = f"{top_line}-{reg_fmt}"
        
        if CAR_PATTERN.match(final_string.replace(serial_letters, serial_letters[0])):
            return final_string

    # Format 2: Motorcycle (DD-LN-XXX.XX)
    if len(candidate_raw) == 9 and candidate_raw[3].isdigit():
        serial_letter = candidate_raw[2]
        serial_digit = candidate_raw[3]
        reg_number_raw = candidate_raw[4:9]
        
        top_line = f"{prov_code}-{serial_letter}{serial_digit}"
        reg_fmt = f"{reg_number_raw[:-2]}.{reg_number_raw[-2:]}"
        final_string = f"{top_line}-{reg_fmt}"
        
        if MOTO_PATTERN.match(final_string):
            return final_string

    # Format 3: Standard car (DD-L-XXX.XX) or short motorcycle
    if len(candidate_raw) == 8:
        # Try standard car format FIRST
        serial_letter = candidate_raw[2]
        reg_number_raw = candidate_raw[3:8]
        
        top_line = f"{prov_code}-{serial_letter}"
        reg_fmt = f"{reg_number_raw[:-2]}.{reg_number_raw[-2:]}"
        final_string = f"{top_line}-{reg_fmt}"

        if CAR_PATTERN.match(final_string):
            # Smart correction: 80 (police) vs 30 (Hanoi) confusion
            if prov_code == "80" and serial_letter in ["A", "B", "C", "D", "E", "F", "G", "H", "K"]:
                alt_string = f"30-{serial_letter}-{reg_fmt}"
                if CAR_PATTERN.match(alt_string):
                    return alt_string
            return final_string
        
        # If car format failed, try motorcycle with 4-digit registration
        if candidate_raw[3].isdigit() and candidate_raw[3] in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
            serial_letter = candidate_raw[2]
            serial_digit = candidate_raw[3]
            reg_number_raw = candidate_raw[4:8]
            
            if len(reg_number_raw) == 4:
                top_line = f"{prov_code}-{serial_letter}{serial_digit}"
                reg_fmt_2 = f"{reg_number_raw[:2]}.{reg_number_raw[2:]}"
                final_string_2 = f"{top_line}-{reg_fmt_2}"
                
                # Only return if it matches the motorcycle pattern
                if MOTO_PATTERN.match(final_string_2):
                    return final_string_2

    # Format 4: Old car format (DD-L-NNNN)
    if len(candidate_raw) == 7:
        serial_letter = candidate_raw[2]
        reg_number_raw = candidate_raw[3:7]
        
        top_line = f"{prov_code}-{serial_letter}"
        final_string = f"{top_line}-{reg_number_raw}"
        
        if reg_number_raw.isdigit() and len(reg_number_raw) == 4:
            # Smart correction for old format
            if prov_code == "80" and serial_letter in ["A", "B", "C", "D", "E", "F", "G", "H", "K", "M"]:
                alt_string = f"30-{serial_letter}-{reg_number_raw}"
                return alt_string
            return final_string

    return ""

# ============================================================================
# 5. PLATE RECOGNITION FUNCTION
# ============================================================================

def recognize_plate(plate_crop, return_image=False):
    """
    Main recognition pipeline: preprocess → OCR → validate.
    Returns:
        - formatted_plate (str): Recognized text or ""
        - plate_input (numpy.ndarray): Preprocessed image (only if return_image=True)
    """
    # Preprocess the cropped plate image
    plate_input = preprocess_plate(plate_crop)
    if plate_input is None:
        if return_image:
            return "", None
        return ""

    try:
        # Run EasyOCR on preprocessed image
        ocr_results = reader.readtext(plate_input, detail=0, allowlist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-')
        
        if not ocr_results:
            print("DEBUG: EasyOCR found NO text.")
            if return_image:
                return "", plate_input
            return ""

        # Concatenate all OCR segments (handles multi-line plates)
        print(f"DEBUG: OCR List: {ocr_results}")
        raw_combined_text = "".join(ocr_results)
        print(f"DEBUG: Raw OCR text: '{raw_combined_text}'")

        # Apply correction and validation
        formatted_plate = correct_plate_format(raw_combined_text)
        
        if return_image:
            return formatted_plate, plate_input
        return formatted_plate

    except Exception:
        pass

    if return_image:
        return "", plate_input
    return ""
