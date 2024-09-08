import os
import shutil

def copy_and_continue_numbering(folder_path):
    # Get the list of all files in the folder
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])

    # Filter out the numbered files in the correct format
    jpg_files = [f for f in files if f[:-4].isdigit() and f.endswith('.jpg')]

    if not jpg_files:
        print("No valid .jpg files found.")
        return

    # Determine the next starting number by parsing the last file's name
    last_file = jpg_files[-1]
    next_number = int(last_file[:-4]) + 1

    for i, file_name in enumerate(jpg_files):
        old_file_path = os.path.join(folder_path, file_name)
        new_file_name = f"{next_number + i:03d}.jpg"  # Keep 3-digit format
        new_file_path = os.path.join(folder_path, new_file_name)
        
        # Copy and rename the file
        shutil.copy(old_file_path, new_file_path)
        print(f"Copied {file_name} to {new_file_name}")

# Example usage
folder_path = './Maxwell'
copy_and_continue_numbering(folder_path)
