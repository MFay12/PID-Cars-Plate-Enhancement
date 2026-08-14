import tkinter as tk
from tkinter import filedialog
import glob
import os

def select_folder():
    """Opens the dialog and returns the folder path and the list of images inside it."""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    folder_path = filedialog.askdirectory(
        title="Select the folder with the vehicle images"
    )
    
    if not folder_path:
        return None, []
        
    # Search for common image extensions in the selected folder
    search_types = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    files = []
    for ext in search_types:
        files.extend(glob.glob(os.path.join(folder_path, ext)))
        
    return folder_path, files
