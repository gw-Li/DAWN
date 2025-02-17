import glob
import os

# List of folders
folder_list = ['20231003','20231008','20231013','20231018','20231023','20231028','20231102','20231107','20231112','20231117','20231122','20231127']
# Parent folder
parent_folder = '/mnt/gfs01/PUB/S2S/V2023-07/Operational/'

# File patterns to delete
patterns = ['*_T2MAX_*', '*_T2MIN_*', '*_PRAVG_*', '*_ASWDNS_*']

# Loop through each folder and delete files matching the patterns
for folder in folder_list:
    for pattern in patterns:
        # Construct full pattern path
        full_pattern = os.path.join(parent_folder, folder, pattern)
        
        # Find all files matching the pattern
        files = glob.glob(full_pattern)
        
        # Delete each file
        for file in files:
            try:
                os.remove(file)  # Using os.remove() which is equivalent to 'rm -f'
            except OSError as e:
                print(f"Error deleting file {file}: {e}")

print("File deletion complete.")


# Parent
