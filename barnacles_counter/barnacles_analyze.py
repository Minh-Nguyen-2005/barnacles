# Author: Minh Nguyen

import cv2
import numpy as np

def analyze_and_count_barnacles(image_path: str):
    """
    Identifies, counts, and visualizes barnacles in a cropped image.

    Args:
        image_path (str): Path to the cropped input image (e.g., 'masked_img1.png').

    Returns:
        A tuple containing:
        - final_count (int): The number of barnacles detected.
        - output_image (np.array): The original image with detected barnacles outlined.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image from {image_path}")
        return 0, None

    # --- Image Pre-processing ---
    # Convert to grayscale, as color is not needed for shape detection.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply a blur to reduce noise and smooth edges.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use adaptive thresholding to create a binary image (black and white).
    # This is great for handling different lighting conditions across the image.
    binary_image = cv2.adaptiveThreshold(
        blurred, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        11, # Block size
        2   # Constant to subtract from the mean
    )
    cv2.imwrite("debug_binary_image.png", binary_image) # Save for inspection

    # --- Contour Detection ---
    # Find all distinct shapes in the binary image.
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- Contour Filtering (The "Intelligence") ---
    filtered_contours = []
    for c in contours:
        # 1. Filter by Area: Remove things that are too small (noise) or too large (clumps).
        area = cv2.contourArea(c)
        if area < 30 or area > 1500: # These values need tuning!
            continue

        # 2. Filter by Shape (Circularity): Keep things that are roughly circular.
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity < 0.5: # A perfect circle has a circularity of 1.
            continue
            
        # If the contour passes all filters, it's likely a barnacle.
        filtered_contours.append(c)

    # --- Counting & Visualization ---
    final_count = len(filtered_contours)
    
    # Create a copy of the original image to draw on.
    output_image = image.copy()
    
    # Draw the outlines of the detected barnacles in a bright color (e.g., green).
    cv2.drawContours(output_image, filtered_contours, -1, (0, 255, 0), 2)
    
    # Put the final count on the image.
    cv2.putText(
        output_image, 
        f"Detected Barnacles: {final_count}", 
        (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, (0, 0, 255), 2
    )

    return final_count, output_image