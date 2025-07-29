# Author: Minh Nguyen

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


def get_line_intersection(line1, line2):
    """Calculates the intersection point of two lines."""
    x1, y1, x2, y2 = line1[0]
    x3, y3, x4, y4 = line2[0]
    A1, B1 = y2 - y1, x1 - x2
    C1 = A1 * x1 + B1 * y1
    A2, B2 = y4 - y3, x3 - x4
    C2 = A2 * x3 + B2 * y3
    determinant = A1 * B2 - A2 * B1
    if determinant == 0:
        return None  # Parallel lines
    intersect_x = (B2 * C1 - B1 * C2) / determinant
    intersect_y = (A1 * C2 - A2 * C1) / determinant
    return (intersect_x, intersect_y)

def merge_lines(lines, is_horizontal=True):
    """Merges close-by line segments into single representative lines."""
    if not lines:
        return []
    sort_index = 1 if is_horizontal else 0
    lines.sort(key=lambda line: line[0][sort_index])
    merged_lines = []
    current_group = [lines[0]]
    for i in range(1, len(lines)):
        avg_pos_group = np.mean([l[0][sort_index] for l in current_group])
        pos_current = lines[i][0][sort_index]
        # Group lines that are close to each other
        if abs(pos_current - avg_pos_group) < 50:
            current_group.append(lines[i])
        else:
            avg_line = np.mean(current_group, axis=0, dtype=np.int32)
            merged_lines.append(avg_line)
            current_group = [lines[i]]
    # Add the last group
    avg_line = np.mean(current_group, axis=0, dtype=np.int32)
    merged_lines.append(avg_line)
    return merged_lines

def crop_inner_square(image_path: str, output_path: str = "cropped_inner_square.png"):
    """
    Crops the inner square of a grid in an image using robust color masking
    in the L*a*b* space and morphological cleaning.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # --- Step 1: Isolate Green using L*a*b* color space ---
    # This is more robust to lighting changes than HSV.
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    a_channel = lab_image[:, :, 1]
    # In the a* channel, lower values are greener. We threshold for values
    # below the midpoint (128), effectively selecting all green tones.
    _, mask = cv2.threshold(a_channel, 120, 255, cv2.THRESH_BINARY_INV)

    # --- Step 2: Morphological Cleaning ---
    # This is the critical step to remove noise and repair the grid lines.
    # Opening removes small noise (from barnacles). A small kernel is key.
    open_kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=2)
    
    # Closing fills gaps in the grid lines caused by shadows/highlights.
    close_kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    
    cv2.imwrite("debug_final_mask.png", mask) # For inspection

    # --- Step 3: Line Detection on the Clean Mask ---
    edges = cv2.Canny(mask, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=100, maxLineGap=40)

    if lines is None:
        print("Could not detect any lines. Check 'debug_final_mask.png'.")
        return

    horizontal_lines, vertical_lines = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        if angle < 45 or angle > 135:
            horizontal_lines.append(line)
        else:
            vertical_lines.append(line)

    # --- Step 4: Merge, Intersect, and Crop ---
    merged_h_lines = merge_lines(horizontal_lines, is_horizontal=True)
    merged_v_lines = merge_lines(vertical_lines, is_horizontal=False)
    
    print(f"Detected and merged into {len(merged_h_lines)} horizontal and {len(merged_v_lines)} vertical lines.")

    if len(merged_h_lines) < 4 or len(merged_v_lines) < 4:
        print("Did not find a complete 4x4 grid. Cropping aborted.")
        return

    merged_h_lines.sort(key=lambda line: np.mean([line[0][1], line[0][3]]))
    merged_v_lines.sort(key=lambda line: np.mean([line[0][0], line[0][2]]))

    # Find inner grid lines (2nd and 3rd lines in the sorted lists)
    h_line1, h_line2 = merged_h_lines[1], merged_h_lines[2]
    v_line1, v_line2 = merged_v_lines[1], merged_v_lines[2]

    # Find the four corner points of the inner square
    top_left = get_line_intersection(h_line1, v_line1)
    top_right = get_line_intersection(h_line1, v_line2)
    bottom_left = get_line_intersection(h_line2, v_line1)
    bottom_right = get_line_intersection(h_line2, v_line2)

    if not all([top_left, top_right, bottom_left, bottom_right]):
        print("Could not find all four inner intersection points. Cropping aborted.")
        return

    corners = np.float32([top_left, top_right, bottom_right, bottom_left])

    # Calculate destination dimensions based on the average length of the detected sides
    width_top = np.linalg.norm(np.array(top_right) - np.array(top_left))
    width_bottom = np.linalg.norm(np.array(bottom_right) - np.array(bottom_left))
    avg_width = int((width_top + width_bottom) / 2.0)

    height_left = np.linalg.norm(np.array(bottom_left) - np.array(top_left))
    height_right = np.linalg.norm(np.array(bottom_right) - np.array(top_right))
    avg_height = int((height_left + height_right) / 2.0)

    dst_points = np.float32([[0, 0], [avg_width - 1, 0], [avg_width - 1, avg_height - 1], [0, avg_height - 1]])

    # Perform the perspective warp on the original color image
    matrix = cv2.getPerspectiveTransform(corners, dst_points)
    warped_image = cv2.warpPerspective(image, matrix, (avg_width, avg_height))

    cv2.imwrite(output_path, warped_image)
    print(f"Successfully cropped and saved the inner square to {output_path}")




def crop_to_inner_frame(img_bgr, debug=False):
    """
    Detects the green square frame by edge detection, approximates it to 4 corners,
    and returns the cropped interior as an RGB image, or None if not found.
    """
    # 1) Grayscale + blur
    gray   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur   = cv2.GaussianBlur(gray, (5,5), 0)

    # 2) Canny edges + closing to fill gaps
    edges  = cv2.Canny(blur, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 3) Find all contours, filter to quadrilaterals not touching the image border
    hImg, wImg = gray.shape
    cnts, _    = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quads = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < 10000:  # skip small blobs
            continue
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            x,y,w,h = cv2.boundingRect(approx)
            # drop any quad that hugs the image edge
            if x < 5 or y < 5 or x+w > wImg-5 or y+h > hImg-5:
                continue
            quads.append((cnt, approx))

    if not quads:
        if debug:
            print("No valid frame quad found")
        return None

    # 4) Choose the largest valid quad
    frame_cnt, frame_approx = max(quads, key=lambda ca: cv2.contourArea(ca[0]))

    # 5) Order corner points (tl, tr, br, bl)
    pts  = frame_approx.reshape(4,2)
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect = np.array([
        pts[np.argmin(s)],
        pts[np.argmin(diff)],
        pts[np.argmax(s)],
        pts[np.argmax(diff)]
    ], dtype="float32")

    # 6) Compute target size
    (tl, tr, br, bl) = rect
    maxW = int(max(np.linalg.norm(br-bl), np.linalg.norm(tr-tl)))
    maxH = int(max(np.linalg.norm(tr-br), np.linalg.norm(tl-bl)))

    # 7) Warp perspective
    dst    = np.array([[0,0], [maxW-1,0], [maxW-1,maxH-1], [0,maxH-1]], dtype="float32")
    M      = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img_bgr, M, (maxW, maxH))

    # 8) Convert to RGB and return
    cropped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    return cropped_rgb



def notebook_show(name, img, w=800, h=600):
    disp = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    plt.figure(figsize=(6,4))
    plt.title(name)
    plt.imshow(disp, cmap='gray' if img.ndim==2 else None)
    plt.axis('off')
    plt.show()

def crop_to_inner_frame_canny(img_bgr, debug=False):
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Close tiny gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 1) Find *all* contours
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    # 2) From all contours, keep only those that approximate to a quadrilateral
    quads = []
    for cnt in cnts:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
        if len(approx)==4 and cv2.contourArea(cnt) > 10000:  # area threshold to skip tiny quads
            quads.append((cnt, approx))

    if not quads:
        if debug: print("No quad contours found—try lowering the area threshold or eps.")
        return None

    # 3) Pick the *largest* quad by area
    frame_cnt, frame_approx = max(quads, key=lambda ca: cv2.contourArea(ca[0]))

    # 4) (Optional) debug‐show which quad you picked
    if debug:
        overlay = img_bgr.copy()
        cv2.drawContours(overlay, [frame_cnt], -1, (255,0,0), 4)
        notebook_show("Picked Frame Contour", overlay)

    # 5) Proceed with the warp using frame_approx
    pts = frame_approx.reshape(4,2)
    s   = pts.sum(axis=1)
    diff= np.diff(pts, axis=1)
    rect= np.array([pts[np.argmin(s)],
                    pts[np.argmin(diff)],
                    pts[np.argmax(s)],
                    pts[np.argmax(diff)]], dtype="float32")

    (tl, tr, br, bl) = rect
    maxW = int(max(np.linalg.norm(br-bl), np.linalg.norm(tr-tl)))
    maxH = int(max(np.linalg.norm(tr-br), np.linalg.norm(tl-bl)))
    dst  = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], dtype="float32")

    M    = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(img_bgr, M, (maxW, maxH))
    cropped_rgb = cv2.cvtColor(warp, cv2.COLOR_BGR2RGB)

    if debug:
        notebook_show("Cropped Int.", warp)

    return cropped_rgb


def debug_crop(img_bgr):
    # 1) convert to HSV
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    print("STEP 1: img shape:", img_bgr.shape)
    
    # show HSV channels
    fig, axs = plt.subplots(1,3, figsize=(12,4))
    for i, name in enumerate(("Hue","Sat","Val")):
        axs[i].imshow(hsv[:,:,i], cmap='gray')
        axs[i].set_title(name)
        axs[i].axis('off')
    plt.show()
    
    # 2) threshold for green
    lower = np.array([40, 60, 60])
    upper = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    print("STEP 2: raw mask unique vals:", np.unique(mask))
    plt.imshow(mask, cmap='gray'); plt.title("green_mask"); plt.axis('off'); plt.show()
    
    # 3) morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    plt.imshow(mask_clean, cmap='gray'); plt.title("after closing"); plt.axis('off'); plt.show()
    
    # 4) find contours
    cnts, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("STEP 4: #contours found:", len(cnts))
    if not cnts:
        print("→ no contours at all, need to widen HSV range")
        return None
    
    # overlay the largest contour
    frame = max(cnts, key=cv2.contourArea)
    overlay = img_bgr.copy()
    cv2.drawContours(overlay, [frame], -1, (0,0,255), 3)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("largest contour overlay"); plt.axis('off'); plt.show()
    
    # 5) approx to polygon
    peri = cv2.arcLength(frame, True)
    approx = cv2.approxPolyDP(frame, 0.02*peri, True)
    print("STEP 5: approx vertices:", len(approx))
    if len(approx)!=4:
        print("→ not 4 points, maybe shape is noisy or thresholds off")
        return None
    
    # if we got here, we can warp—just show the approx pts:
    pts = approx.reshape(4,2)
    print("STEP 5 pts:\n", pts)
    return True


def crop_to_inner_frame_hsv(img_bgr, debug=False):
    """
    Finds the green square frame by color thresholding in HSV space,
    approximates it to 4 corners, and returns the warped interior as RGB.
    """
    # 1) Convert to HSV
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # 2) Threshold for the green wire (tweak these if needed)
    lower_green = np.array([50, 100,  50])
    upper_green = np.array([75, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # 3) Clean up: open then close
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    if debug:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1); plt.title("Raw Green Mask");  plt.imshow(mask, cmap='gray'); plt.axis('off')
        plt.subplot(1,2,2); plt.title("Cleaned Mask");    plt.imshow(mask, cmap='gray'); plt.axis('off')
        plt.show()
    
    # 4) Find contours, keep only 4-corner quads above area threshold
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quads = []
    h, w = mask.shape
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < 5000:
            continue
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            quads.append(approx)
    if not quads:
        if debug: print("No quads found in HSV mask.")
        return None
    
    # 5) Pick the largest quad by area
    quad = max(quads, key=lambda a: cv2.contourArea(a))
    pts  = quad.reshape(4,2).astype("float32")
    
    # 6) Order points (tl, tr, br, bl)
    s    = pts.sum(axis=1); diff = np.diff(pts, axis=1)
    rect = np.array([
        pts[np.argmin(s)],
        pts[np.argmin(diff)],
        pts[np.argmax(s)],
        pts[np.argmax(diff)]
    ], dtype="float32")
    
    # 7) Compute target size
    (tl, tr, br, bl) = rect
    maxW = int(max(np.linalg.norm(br-bl), np.linalg.norm(tr-tl)))
    maxH = int(max(np.linalg.norm(tr-br), np.linalg.norm(tl-bl)))
    
    # 8) Warp & crop
    dst    = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]],dtype="float32")
    M      = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img_bgr, M, (maxW, maxH))
    
    # 9) Convert to RGB and return
    return cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
