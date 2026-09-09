"""通过 FFmpeg 将 OpenCV BGR 帧直接编码为 H.264 MP4。"""

import subprocess
from pathlib import Path
from types import TracebackType

import numpy as np
from numpy.typing import NDArray


class H264VideoWriter:
    """把固定尺寸的 BGR uint8 帧写入 FFmpeg 的标准输入。"""

    def __init__(
        self,
        output_path: Path,
        fps: float,
        frame_size: tuple[int, int],
    ) -> None:
        width, height = frame_size
        if fps <= 0:
            raise ValueError("视频帧率必须大于 0。")
        if width <= 0 or height <= 0:
            raise ValueError("视频分辨率必须大于 0。")

        self.output_path = output_path
        self.frame_size = frame_size
        self._closed = False
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except FileNotFoundError as error:
            raise RuntimeError("找不到 ffmpeg，请确认当前 Conda 环境已安装 ffmpeg。") from error
        if self._process.stdin is None or self._process.stderr is None:
            self._process.kill()
            raise RuntimeError("无法建立 FFmpeg 视频编码管道。")

    def write(self, frame: NDArray[np.uint8]) -> None:
        """写入一帧 BGR24 图像。"""
        if self._closed:
            raise RuntimeError("视频写入器已经关闭。")

        width, height = self.frame_size
        if frame.dtype != np.uint8 or frame.shape != (height, width, 3):
            raise ValueError(
                f"视频帧必须是 {(height, width, 3)} 的 uint8 BGR 图像，"
                f"实际为 shape={frame.shape}、dtype={frame.dtype}。"
            )

        if self._process.poll() is not None:
            raise RuntimeError(self._failure_message())
        try:
            self._process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except (BrokenPipeError, OSError) as error:
            raise RuntimeError(self._failure_message()) from error

    def release(self) -> None:
        """关闭输入并等待 FFmpeg 完成 MP4 封装。"""
        if self._closed:
            return
        self._closed = True
        try:
            self._process.stdin.close()
        except BrokenPipeError:
            pass
        stderr = self._process.stderr.read().decode("utf-8", errors="replace").strip()
        return_code = self._process.wait()
        self._process.stderr.close()
        if return_code != 0:
            detail = stderr or f"退出码 {return_code}"
            raise RuntimeError(f"FFmpeg 无法生成视频 {self.output_path}：{detail}")

    def _failure_message(self) -> str:
        stderr = self._process.stderr.read().decode("utf-8", errors="replace").strip()
        detail = stderr or f"退出码 {self._process.returncode}"
        return f"FFmpeg 无法生成视频 {self.output_path}：{detail}"

    def __enter__(self) -> "H264VideoWriter":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()


def create_video_writer(
    output_path: Path,
    fps: float,
    frame_size: tuple[int, int],
) -> H264VideoWriter:
    """创建直接输出 H.264/yuv420p MP4 的统一视频写入器。"""
    return H264VideoWriter(output_path, fps, frame_size)
