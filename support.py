# support.py
import subprocess

def get_window_linux(name_pattern: str):
    # 1) try xdotool
    try:
        out = subprocess.check_output(
            ["xdotool", "search", "--onlyvisible", "--name", name_pattern],
            stderr=subprocess.DEVNULL
        ).decode().strip().split()
        if out:
            wid = out[0]
            # use xwininfo have shape
            info = subprocess.check_output(
                ["xwininfo", "-id", wid]
            ).decode().splitlines()
            left = top = width = height = None
            for line in info:
                line = line.strip()
                if line.startswith("Absolute upper-left X:"):
                    left = int(line.split(":")[1])
                elif line.startswith("Absolute upper-left Y:"):
                    top = int(line.split(":")[1])
                elif line.startswith("Width:"):
                    width = int(line.split(":")[1])
                elif line.startswith("Height:"):
                    height = int(line.split(":")[1])
            if None not in (left, top, width, height):
                return (left, top, width, height)
    except subprocess.CalledProcessError:
        pass

    # 2) back to wmctrl
    try:
        lines = subprocess.check_output(
            ["wmctrl", "-lG"],
            stderr=subprocess.DEVNULL
        ).decode().splitlines()
        for L in lines:
            # wmctrl -lG: ID DESKTOP X Y W H HOST TITLE...
            parts = L.split(maxsplit=7)
            if len(parts) >= 8 and name_pattern in parts[7]:
                _, _, x, y, w, h, _, _ = parts
                return (int(x), int(y), int(w), int(h))
    except subprocess.CalledProcessError:
        pass

    return None
