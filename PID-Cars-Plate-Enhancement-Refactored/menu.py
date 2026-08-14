import cv2
import sys
import os
from selector import select_folder
from car_plate import search_car_plate
from motorcycle_plate import search_motorcycle_plate
from character import validate_characters

def main():
    print("\nSelect an option:")
    print("1 - Only Cars")
    print("2 - Only Motorcycles")
    print("3 - Both")
    search_option = input("Number (1, 2 or 3): ")

    folder_path, files = select_folder()
    
    if not folder_path or not files:
        print("No folder attached...")
        sys.exit()

    dest_folder = "./cropped_plates"
    os.makedirs(dest_folder, exist_ok=True)

    print(f"Found {len(files)} images in the folder.")
    counter = 1

    for img_path in files:
        print(f"\nProcessing: {img_path}")
        colored_img = cv2.imread(img_path)
        
        if colored_img is None:
            continue

        gray_img = cv2.cvtColor(colored_img, cv2.COLOR_BGR2GRAY)
        success = False

        # ==========================================
        # CAR SEARCH
        # ==========================================
        if search_option in ['1', '3']:
            debug_img, top_5_cars = search_car_plate(gray_img, colored_img)
            
            for index, candidate in enumerate(top_5_cars):
                score, x, y, width, height = candidate
                cropped_img = gray_img[y:y+height, x:x+width]
                plate_bin, characters = validate_characters(cropped_img)
                
                if characters >= 5:
                    success = True
                    
                    cv2.rectangle(debug_img, (x, y), (x + width, y + height), (0, 255, 0), 3)
                    cv2.putText(debug_img, "WINNER", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    print(f"[SUCCESS] CAR plate found in candidate #{index+1} ({characters} characters).")
                    cv2.imshow("General Debug", debug_img)
                    cv2.imshow("Winner Crop", cropped_img)
                    cv2.imshow("Characters", plate_bin)
                    
                    cv2.imwrite(os.path.join(dest_folder, f"car_plate_{counter}.jpg"), cropped_img)
                    counter += 1
                    break 

            if not success and search_option == '1':
                print("[FAILURE] None of the 5 car candidates passed the test.")
                cv2.imshow("General Debug", debug_img)

        # ==========================================
        # MOTORCYCLE SEARCH
        # ==========================================
        if not success and search_option in ['2', '3']:
            debug_img, top_5_motorcycles = search_motorcycle_plate(gray_img, colored_img)
            
            for index, candidate in enumerate(top_5_motorcycles):
                score, x, y, width, height = candidate
                cropped_img = gray_img[y:y+height, x:x+width]
                plate_bin, characters = validate_characters(cropped_img)
                
                if characters >= 5:
                    success = True
                    cv2.rectangle(debug_img, (x, y), (x + width, y + height), (255, 0, 0), 3)
                    cv2.putText(debug_img, "WINNER", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    
                    print(f"[SUCCESS] MOTORCYCLE plate found in candidate #{index+1} ({characters} characters).")
                    cv2.imshow("General Debug", debug_img)
                    cv2.imshow("Winner Crop", cropped_img)
                    cv2.imshow("Characters", plate_bin)
                    
                    cv2.imwrite(os.path.join(dest_folder, f"motorcycle_plate_{counter}.jpg"), cropped_img)
                    counter += 1
                    break 

            if not success:
                print("[FAILURE] None of the candidates passed the test.")
                cv2.imshow("General Debug", debug_img)

        key = cv2.waitKey(0)
        if key == 27: 
            break
            
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
