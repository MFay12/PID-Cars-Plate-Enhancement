# License Plate Recognizer

This project implements a pipeline to geometrically validate vehicle license plates using images. Developed in Python, the system uses the OpenCV library for image processing, Matplotlib for UI, and Numpy for matrix operations.

## Project Architecture

The code is fully modularized and structured for an English-first environment:

- `menu.py`: The main orchestrator that manages data flow between modules and the visual display of results.
- `selector.py`: Graphical User Interface (GUI) using Tkinter to select image directories from the operating system.
- `heuristic_detector.py`: Shared heuristic detector. It normalizes scale, applies CLAHE, Black Hat, Scharr X, directional morphology, geometric filters, character features extraction, candidate deduplication, and calculates compound scoring.
- `car_plate.py`: Adapter for horizontal plates. It preserves the repository output format `(debug_image, top_5_candidates)`.
- `motorcycle_plate.py`: Adapter for motorcycle plates using the same output format.
- `character.py`: Receives the cropped plate matrix, applies inverted Otsu binarization, and counts structural entities that correspond to valid characters.

## Technologies Used

- **Python 3.13.13**
- **OpenCV (`cv2`)**: Main library for the entire pipeline.
- **NumPy**: For efficient array operations, matrix manipulation, and defining structural kernels.
- **Tkinter**: Standard Python library for the file selection interface.

## How to Run

1. Ensure Python is installed on your system.
2. Install the required dependencies by running the following command in your terminal:
   ```bash
   pip install opencv-python numpy
   ```
3. Execute the main orchestrator:
   ```bash
   python menu.py
   ```
4. A window will pop up prompting you to select a directory containing the vehicle images (Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`).
5. After selection, choose the scope (Cars, Motorcycles, or Both) in the terminal. The algorithm will perform the analysis and display the OpenCV processing stages (General Debug bounding boxes, Winner Crop, and Binarized Characters).
6. Press any key to move to the next image, or press `Esc` on an image window to stop processing. Output images will be saved in the `./cropped_plates` folder.
