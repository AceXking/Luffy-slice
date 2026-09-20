"""视频处理工具 - 入口。

用法示例:
    python main.py probe "demo.mp4"
"""

import sys

from video_processor import ffmpeg_utils


def main(argv: list[str]) -> int:
    if not argv:
        print("用法: python main.py <command> [args]")
        print("可用命令:")
        print("  probe <video>   读取视频元信息")
        return 1

    command = argv[0]
    rest = argv[1:]

    if command == "probe":
        if not rest:
            print("请指定视频文件路径")
            return 1
        info = ffmpeg_utils.probe(rest[0])
        print(info)
        return 0

    print(f"未知命令: {command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
