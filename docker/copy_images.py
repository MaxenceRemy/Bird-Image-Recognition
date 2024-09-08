import os
import shutil
from pathlib import Path

def copy_files_to_new_folder(source_folder, new_folder_name):
    # Get the absolute path of the source folder
    source_folder_path = Path(source_folder).resolve()
    
    # Ensure the source folder exists
    if not source_folder_path.is_dir():
        print(f"Source folder '{source_folder}' does not exist.")
        return

    # Create the destination folder path next to the source folder
    destination_folder_path = source_folder_path.parent / new_folder_name

    # Create the new folder if it doesn't exist
    destination_folder_path.mkdir(exist_ok=True)

    # Copy all files from source to destination
    for item in source_folder_path.iterdir():
        # Check if the item is a file (not a subfolder)
        if item.is_file():
            shutil.copy(item, destination_folder_path)

    print(f"All files from '{source_folder}' have been copied to '{destination_folder_path}'.")

# Example usage
source_folder = "./Abbotts babbler"  # Replace with your actual source folder
new_folder_name = "./Max"  # Replace with the desired folder name

copy_files_to_new_folder(source_folder, new_folder_name)