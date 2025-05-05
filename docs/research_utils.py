import os

def fix_wsl_paths(path: str) -> str:
    """Convert Windows paths to WSL paths if necessary."""
    # If path already starts with /mnt/, assume it's correct WSL format
    if path.startswith('/mnt/'):
        return path
    # Otherwise, attempt conversion from Windows format
    path = path.replace("\\", os.sep)
    drive_parts = path.split(os.sep)
    if len(drive_parts) > 0 and len(drive_parts[0]) > 1 and drive_parts[0][1] == ':':
        drive_letter = drive_parts[0][0].lower()
        # Reconstruct path starting from /mnt/<drive_letter>/...
        wsl_path = f'/mnt/{drive_letter}/' + os.sep.join(drive_parts[1:])
        return wsl_path
    else:
        # If it doesn't look like a Windows path either, return original
        return path
