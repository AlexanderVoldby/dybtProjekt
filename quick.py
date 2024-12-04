import os

folder_path = '/Users/fredmac/Documents/DTU-FredMac/Deep/dybtProjekt/data/no-background/no-background-stanford-cars-synthetic-classwise'

total_files = sum(len(files) for _, _, files in os.walk(folder_path))
print(f"Total number of files: {total_files}")