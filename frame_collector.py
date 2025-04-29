#!/usr/bin/env python3
import os
import sys
import time
import argparse
import subprocess

from mss import mss
import psutil
import h5py
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
if here not in sys.path:
    sys.path.insert(0, here)
try:
    from support import get_window_linux
except ImportError:
    get_window_linux = None

def collect_frames(out_dir, manifest_path, h5_path, args):
    os.makedirs(out_dir, exist_ok=True)

    sct = mss()
    interval = 1.0 / args.fps
    count = 0

    hf = None
    dset = None

    with open(manifest_path, 'w') as mf:
        while args.max is None or count < args.max:
            start = time.time()

            if args.window:
                if get_window_linux is None:
                    print("[ERROR] get_window_linux cant work", file=sys.stderr)
                    break
                geom = get_window_linux(args.window)
                if geom is None:
                    print(f"[WARN] did not detect '{args.window}', jump frame", file=sys.stderr)
                    time.sleep(interval)
                    continue
                left, top, width, height = geom
            else:
                top, left, width, height = args.region

            shot = sct.grab({'left': left, 'top': top, 'width': width, 'height': height})
            img = np.frombuffer(shot.rgb, dtype=np.uint8).reshape((height, width, 3))

            gray = np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype('uint8')

            if hf is None:
                hf = h5py.File(h5_path, "w", libver="latest")
                dset = hf.create_dataset(
                    "frames",
                    shape=(0, height, width),
                    maxshape=(None, height, width),
                    dtype='uint8',
                    chunks=(1, height, width),
                    shuffle=True,
                    compression="gzip",
                    compression_opts=9
                )
                hf.swmr_mode = True
                print(f"[INFO] HDF5 初始化：{h5_path} (/frames)，灰度+gzip+shuffle", flush=True)

            dset.resize((count + 1), axis=0)
            dset[count, :, :] = gray
            hf.flush()
            mf.write(f"{count}\n")
            mf.flush()

            count += 1
            elapsed = time.time() - start
            if elapsed < interval:
                time.sleep(interval - elapsed)

    if hf is not None:
        hf.close()
        print(f"[DONE] total {count} frames to '{h5_path}'", flush=True)
    else:
        print("[WARN] none frame, did not create the h5 file.", flush=True)


def terminate_proc_tree(pid):
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    for child in parent.children(recursive=True):
        try: child.terminate()
        except: pass
    try:
        parent.terminate()
    except: pass


def main():
    parser = argparse.ArgumentParser(
        description="Frame Collector → gray + gzip+shuffle HDF5"
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--window", help="title of window (Linux)")
    grp.add_argument(
        "--region", nargs=4, type=int,
        metavar=('TOP','LEFT','WIDTH','HEIGHT'),
        help="Manually select area: top left width height"
    )
    parser.add_argument("--out",    required=True, help="output folder (for manifest)")
    parser.add_argument("--man",    required=True, help="manifest path")
    parser.add_argument("--h5",     required=True, help=" HDF5 path")
    parser.add_argument("--max",    type=int,   default=None, help="max fps, default inf")
    parser.add_argument("--fps",    type=float, default=5.0,  help="fps, default 5")
    parser.add_argument(
        "--launch-env", action="store_true",
        help="launch env.py"
    )
    parser.add_argument(
        "--env-script", default="env.py",
        help="env path."
    )
    args = parser.parse_args()

    proc = None
    if args.launch_env:
        try:
            proc = subprocess.Popen([sys.executable, args.env_script])
            print(f"[INFO] launch the env '{args.env_script}' (pid={proc.pid})", flush=True)
            time.sleep(3)
        except Exception as e:
            print(f"[ERROR] fail to launch the env: {e}", file=sys.stderr)
            sys.exit(1)

    try:
        collect_frames(args.out, args.man, args.h5, args)
    finally:
        if proc:
            terminate_proc_tree(proc.pid)
            print(f"[INFO] Terminate the env script (root pid={proc.pid})", flush=True)


if __name__ == "__main__":
    main()

