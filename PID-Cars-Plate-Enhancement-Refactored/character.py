import cv2

def validate_characters(cropped_plate_img):
    """Analyzes the contours of the cropped plate to count valid characters."""
    if cropped_plate_img is None:
        return None, 0

    _, plate_bin = cv2.threshold(cropped_plate_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    plate_contours, _ = cv2.findContours(plate_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    found_characters = 0
    for c in plate_contours:
        xc, yc, wc, hc = cv2.boundingRect(c)
        c_ratio = wc / float(hc)
        
        if 0.2 < c_ratio < 0.9 and (wc * hc) > 40:
            found_characters += 1
            
    return plate_bin, found_characters
