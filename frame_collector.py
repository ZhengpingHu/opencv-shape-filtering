#!/usr/bin/env python3
"""
frame_collector.py

- 保留原有 window/region 检测逻辑（support.get_window_linux）
- 将每帧转为灰度，不再生成 PNG
- 以 gzip+shuffle 压缩（compression_opts=9）写入 HDF5 (/frames)
- 记录每帧索引到 manifest.txt
- 捕获结束后，终止通过 --launch-env 启动的脚本及其子进程
"""

import os
import sys
import time
import argparse
import subprocess

from mss import mss
import psutil
import h5py
import numpy as np

# support.py 同目录下，用于 window 检测
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

            # —— 窗口或区域模式不变 —— 
            if args.window:
                if get_window_linux is None:
                    print("[ERROR] get_window_linux 不可用，无法检测窗口", file=sys.stderr)
                    break
                geom = get_window_linux(args.window)
                if geom is None:
                    print(f"[WARN] 未检测到窗口 '{args.window}'，跳过本帧", file=sys.stderr)
                    time.sleep(interval)
                    continue
                left, top, width, height = geom
            else:
                top, left, width, height = args.region

            # 抓屏
            shot = sct.grab({'left': left, 'top': top, 'width': width, 'height': height})
            img = np.frombuffer(shot.rgb, dtype=np.uint8).reshape((height, width, 3))

            # 转为灰度：Y = 0.299 R + 0.587 G + 0.114 B
            gray = np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype('uint8')

            # 第一次初始化 HDF5
            if hf is None:
                hf = h5py.File(h5_path, "w", libver="latest")
                # shape=(0,H,W)，maxshape 第一维无限
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

            # 追加一帧
            dset.resize((count + 1), axis=0)
            dset[count, :, :] = gray
            hf.flush()   # 保证 SWMR 读者可见

            # 写 manifest（记录帧索引）
            mf.write(f"{count}\n")
            mf.flush()

            count += 1

            # 控制帧率
            elapsed = time.time() - start
            if elapsed < interval:
                time.sleep(interval - elapsed)

    # 关闭 HDF5
    if hf is not None:
        hf.close()
        print(f"[DONE] 共写入 {count} 帧 到 '{h5_path}'", flush=True)
    else:
        print("[WARN] 未写入任何帧，HDF5 未创建。", flush=True)


def terminate_proc_tree(pid):
    """终止指定 PID 及其所有子进程"""
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
        description="Frame Collector → 灰度+gzip+shuffle HDF5，不生成 PNG"
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--window", help="窗口标题匹配串 (Linux)")
    grp.add_argument(
        "--region", nargs=4, type=int,
        metavar=('TOP','LEFT','WIDTH','HEIGHT'),
        help="手动截屏区域：top left width height"
    )
    parser.add_argument("--out",    required=True, help="输出目录 (用于 manifest)")
    parser.add_argument("--man",    required=True, help="manifest 文件路径")
    parser.add_argument("--h5",     required=True, help="输出 HDF5 文件路径")
    parser.add_argument("--max",    type=int,   default=None, help="最大帧数（默认不限）")
    parser.add_argument("--fps",    type=float, default=5.0,  help="帧率，默认 5")
    parser.add_argument(
        "--launch-env", action="store_true",
        help="启动 env.py（或自定义脚本）"
    )
    parser.add_argument(
        "--env-script", default="env.py",
        help="环境脚本路径 (默认: env.py)"
    )
    args = parser.parse_args()

    proc = None
    if args.launch_env:
        try:
            proc = subprocess.Popen([sys.executable, args.env_script])
            print(f"[INFO] 已启动环境脚本 '{args.env_script}' (pid={proc.pid})", flush=True)
            time.sleep(3)
        except Exception as e:
            print(f"[ERROR] 启动环境脚本失败：{e}", file=sys.stderr)
            sys.exit(1)

    try:
        collect_frames(args.out, args.man, args.h5, args)
    finally:
        if proc:
            terminate_proc_tree(proc.pid)
            print(f"[INFO] 已终止环境脚本及其子进程 (root pid={proc.pid})", flush=True)


if __name__ == "__main__":
    main()

