import os

current_file_name = os.path.basename(__file__)
print(f"Current file name: {current_file_name}")

# Get file name without extension
filename = os.path.splitext(current_file_name)[0]
print(f"File name without extension: {filename}")

# Get file name without extension straight forward
filename = os.path.splitext(os.path.basename(__file__))[0]