import os
import re

def get_missing_groups(folder_path):
    files = os.listdir(folder_path)
    groups = set()

    for f in files:
        nums = re.findall(r'\d+', f)
        if len(nums) >= 2:
            groups.add(int(nums[0]))

    groups = sorted(groups)

    missing = []
    for i in range(groups[0], groups[-1]):
        if i not in groups:
            missing.append(i)

    return missing, groups


intensity_folder = "training_set/intensity/npy"
phase_folder = "training_set/phase/npy"

missing_intensity, groups_intensity = get_missing_groups(intensity_folder)
missing_phase, groups_phase = get_missing_groups(phase_folder)

print("Missing groups in intensity:", missing_intensity)
print("Missing groups in phase:", missing_phase)

print("\nAre missing groups identical?",
      missing_intensity == missing_phase)
