License Plate Recognizer

This project implements a pipeline to geometrically validate a vehicle license plates using images. Developed in Python, the system uses the OpenCV library for image processing, Matplotlib for UI and Numpy for different calcs.

## Project Architecture

The code is modularized:

- `menu.py`: The main orchestrator that manages data flow between modules and the visual display of results.
- `Seletor.py`: Graphical User Interface (GUI) using Tkinter select image files from the operating system.
- `DetectorHeuristico.py`: Shared heuristic detector. It normalizes scale,
  applies CLAHE, Black Hat, Scharr, directional morphology, geometric filters,
  character features, candidate deduplication and compound scoring.
- `PlacaCarro.py`: Adapter for horizontal plates. It preserves the repository
  output format `(debug_image, top_5_candidates)`.
- `PlacaMoto.py`: Adapter for motorcycle plates with the same output format.
- `Caractere.py`: Receives the cropped plate matrix, applies inverted Otsu binarization, and counts structural entities that correspond to valid characters.

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
4. A window will pop up prompting you to select a car image (Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`).
5. After selection, the algorithm will perform the analysis and display the OpenCV processing stages (Bounding Box, Cropped Matrix, and Letter Binarization).
6. Press `Esc` on an image window to stop processing.
