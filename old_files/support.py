import subprocess

def get_window_linux(window_name: str):
    try:
        # get window by keyword
        cmd_search = ["xdotool", "search", "--onlyvisible", "--name", window_name]
        window_ids = subprocess.check_output(cmd_search).decode("utf-8").strip().split("\n")
        if not window_ids or window_ids[0].strip() == "":
            print(f"[get_window_linux] No window found with name: {window_name}")
            return None
        window_id = window_ids[0].strip()

        # get the shape info of window by xwininfo
        cmd_info = ["xwininfo", "-id", window_id]
        info_output = subprocess.check_output(cmd_info).decode("utf-8").strip().split("\n")

        left, top, width, height = None, None, None, None
        for line in info_output:
            line = line.strip()
            if line.startswith("Absolute upper-left X:"):
                left = int(line.split(":")[1])
            elif line.startswith("Absolute upper-left Y:"):
                top = int(line.split(":")[1])
            elif line.startswith("Width:"):
                width = int(line.split(":")[1])
            elif line.startswith("Height:"):
                height = int(line.split(":")[1])

        if None in [left, top, width, height]:
            print("[get_window_linux] Failed to parse xwininfo output.")
            return None
        return (left, top, width, height)

    except subprocess.CalledProcessError as e:
        print(f"[get_window_linux] Error: {e}")
        return None
