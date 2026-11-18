"""Opt-in second-person observations; never an automatic disposition.

Only an explicitly provided, existing, trusted local PyTorch weight is loaded.
The optional backend runs in its own CPU process with Python socket connections
disabled and a temporary Ultralytics configuration. This is not a sandbox for
untrusted weights. The application process does not change its sockets,
environment, runtime configuration, or primary detections.
"""

import hashlib
import math
import multiprocessing
import re
import threading
from pathlib import Path

import numpy as np

from core.person_identity_cv import compare_person_observation


_MAX_PIXELS = 4096 * 2304
_TIMEOUT_SECONDS = 120
_SHA256 = re.compile(r"[a-fA-F0-9]{64}\Z")


class PersonCrosscheckError(RuntimeError):
    """A requested second observation is unavailable, not a negative finding."""


def _file_sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _validate_frame(frame):
    if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3
            or frame.shape[2] != 3 or not 0 < frame.shape[0] * frame.shape[1] <= _MAX_PIXELS):
        raise PersonCrosscheckError("人员二检输入必须为有界的原始 BGR8 图像")


def _views(width, height):
    """Full image plus 2x2 tiles with >=20% overlap, including odd edge pixels."""
    tw, th = math.ceil(width / 1.8), math.ceil(height / 1.8)
    regions = [(0, 0, width, height)]
    for y in (0, height - th):
        for x in (0, width - tw):
            region = (x, y, x + tw, y + th)
            if region not in regions:
                regions.append(region)
    return regions


def _predict_full_and_tiles(model, frame):
    _validate_frame(frame)
    height, width = frame.shape[:2]
    detections = []
    for x, y, right, bottom in _views(width, height):
        tile = np.ascontiguousarray(frame[y:bottom, x:right])
        predictions = model.predict(tile, classes=[0], conf=0.25, imgsz=960,
                                    device="cpu", verbose=False, save=False, stream=False)
        if len(predictions) != 1 or predictions[0].boxes is None:
            raise PersonCrosscheckError("人员二检未返回完整单帧结果")
        boxes = predictions[0].boxes
        xyxy = np.asarray(boxes.xyxy.detach().cpu().numpy(), dtype=float)
        scores = np.asarray(boxes.conf.detach().cpu().numpy(), dtype=float)
        classes = np.asarray(boxes.cls.detach().cpu().numpy(), dtype=float)
        if (xyxy.shape != (len(scores), 4) or scores.ndim != 1 or classes.shape != scores.shape
                or not np.isfinite(xyxy).all() or not np.isfinite(scores).all()
                or not np.isfinite(classes).all() or np.any(classes != 0)
                or np.any((scores < 0.25) | (scores > 1))):
            raise PersonCrosscheckError("人员二检坐标、类别或置信度不合法")
        for raw, confidence in zip(xyxy, scores, strict=True):
            x1, y1, x2, y2 = map(float, raw)
            if not (0 <= x1 < x2 <= right - x and 0 <= y1 < y2 <= bottom - y):
                raise PersonCrosscheckError("人员二检框不在其实际观察图像内")
            detections.append({"class": "person", "confidence": float(confidence),
                               "bbox": {"x": x1 + x, "y": y1 + y, "w": x2 - x1, "h": y2 - y1}})
    detections.sort(key=lambda row: (-row["confidence"], *row["bbox"].values()))
    return detections


def _deny_network(*_args, **_kwargs):
    raise OSError("network disabled in local person crosscheck worker")


def _worker(connection, weights, expected_sha):
    # Deliberately local to a spawn child: do not patch the application's sockets
    # or overwrite its environment/configuration when the optional mode is used.
    import contextlib
    import os
    import socket
    import tempfile

    socket.socket.connect = _deny_network
    socket.socket.connect_ex = _deny_network
    socket.socket.sendto = _deny_network
    socket.create_connection = _deny_network
    socket.getaddrinfo = _deny_network
    try:
        with tempfile.TemporaryDirectory(prefix="person_crosscheck_") as temporary:
            os.environ["YOLO_CONFIG_DIR"] = temporary
            os.environ["YOLO_AUTOINSTALL"] = "false"
            os.environ["YOLO_OFFLINE"] = "true"
            os.environ["WANDB_MODE"] = "disabled"
            with open(os.devnull, "w", encoding="utf-8") as quiet:
                with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                    import ultralytics
                    from ultralytics import YOLO

                    path = Path(weights)
                    if not path.is_file() or _file_sha(path) != expected_sha:
                        raise PersonCrosscheckError("model_changed_before_load")
                    model = YOLO(str(path), task="detect")
                    if model.names.get(0) != "person" or _file_sha(path) != expected_sha:
                        raise PersonCrosscheckError("model_class_or_artifact_mismatch")
                    connection.send({"status": "ready", "ultralytics_version": ultralytics.__version__})
                    while True:
                        request = connection.recv()
                        if request is None:
                            break
                        frame = request
                        _validate_frame(frame)
                        digest = hashlib.sha256(np.ascontiguousarray(frame)).hexdigest()
                        result = _predict_full_and_tiles(model, frame)
                        connection.send({"status": "ok", "frame_sha256": digest,
                                         "model_sha256": expected_sha, "full_frame_covered": True,
                                         "detections": result})
    except (EOFError, BrokenPipeError):
        pass
    except Exception:
        # Error strings from third-party packages may contain local paths or
        # configuration contents; only the fixed failure status crosses the pipe.
        try:
            connection.send({"status": "unavailable", "reason": "secondary_worker_failed"})
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        connection.close()


class PersonCrosschecker:
    """One persistent CPU worker; all observations remain human-review only."""

    def __init__(self, weights, primary_model_sha256):
        self._connection = None
        self._process = None
        if not isinstance(primary_model_sha256, str) or not _SHA256.fullmatch(primary_model_sha256):
            raise PersonCrosscheckError("缺少主检测模型 SHA-256，不能独立记录二检来源")
        try:
            requested = Path(weights)
            if requested.as_posix().startswith("//"):
                raise ValueError("network/share paths are not accepted for weights")
            path = requested.resolve(strict=True)
        except (OSError, TypeError, ValueError) as exc:
            raise PersonCrosscheckError("人员二检必须指定已有本地 .pt 模型，不接受下载名称") from exc
        if path.as_posix().startswith("//") or not path.is_file() or path.suffix.lower() != ".pt":
            raise PersonCrosscheckError("人员二检仅接受已有可信本地 .pt 模型")
        self._primary_sha = primary_model_sha256.lower()
        self._model_sha = _file_sha(path)
        if self._model_sha == self._primary_sha:
            raise PersonCrosscheckError("人员二检模型与主模型相同，不能当作第二权重观察")
        self._metadata = {"method": "local_second_person_detector", "model_sha256": self._model_sha,
                          "primary_model_sha256": self._primary_sha, "backend": "ultralytics_cpu",
                          "views": "full_frame_plus_2x2_overlap20", "confidence": 0.25, "input_size": 960,
                          "network": "blocked_in_isolated_worker", "experimental": True,
                          "needs_review": True, "automatic_disposition_permitted": False,
                          "independence": "different_weights_do_not_prove_independent_errors"}
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        self._connection = parent
        self._process = context.Process(target=_worker, args=(child, str(path), self._model_sha), daemon=True)
        try:
            self._process.start()
            child.close()
            ready = self._receive()
            if ready.get("status") != "ready" or not isinstance(ready.get("ultralytics_version"), str):
                raise PersonCrosscheckError("无法加载人员二检模型／依赖或核对 person 类别；未回退其他模型")
            self._metadata["ultralytics_version"] = ready["ultralytics_version"]
        except BaseException:
            child.close()
            self.close()
            raise

    @property
    def report_metadata(self):
        return dict(self._metadata)

    def _receive(self):
        try:
            if self._connection is None or not self._connection.poll(_TIMEOUT_SECONDS):
                raise PersonCrosscheckError("人员二检子进程超时；本次观察不可用")
            result = self._connection.recv()
            if not isinstance(result, dict):
                raise PersonCrosscheckError("人员二检子进程返回内容无效")
            return result
        except (EOFError, BrokenPipeError, OSError) as exc:
            raise PersonCrosscheckError("人员二检子进程连接失败；本次观察不可用") from exc

    def observe(self, frame, candidate_bboxes):
        _validate_frame(frame)
        digest = hashlib.sha256(np.ascontiguousarray(frame)).hexdigest()
        height, width = frame.shape[:2]
        try:
            if self._connection is None:
                raise PersonCrosscheckError("人员二检进程已关闭")
            self._send_frame(np.ascontiguousarray(frame))
            secondary = self._receive()
            if secondary.get("status") == "ok" and secondary.get("model_sha256") != self._model_sha:
                secondary = {"status": "unavailable", "reason": "unexpected_secondary_model"}
            if secondary.get("status") != "ok":
                self.close()
        except (PersonCrosscheckError, EOFError, BrokenPipeError, OSError):
            self.close()
            secondary = {"status": "unavailable"}
        observations = []
        for candidate in candidate_bboxes:
            comparison = compare_person_observation(
                candidate, frame_sha256=digest, frame_size={"width": width, "height": height},
                primary_model_sha256=self._primary_sha, secondary=secondary)
            # Even unavailable comparisons carry the caller's exact original
            # candidate and frame provenance; neither can be used as a verdict.
            comparison.update(candidate_bbox=dict(candidate), frame_sha256=digest)
            observations.append(comparison)
        return observations

    def _send_frame(self, frame):
        # Windows Pipe.send can block if a failed/hung child stops reading a
        # large image. Bound sending as well as receiving; the daemon thread
        # never outlives a deliberately terminated worker's pipe indefinitely.
        connection = self._connection
        finished = threading.Event()
        errors = []

        def send():
            try:
                connection.send(frame)
            except Exception:
                errors.append(True)
            finally:
                finished.set()

        sender = threading.Thread(target=send, daemon=True, name="person-crosscheck-send")
        sender.start()
        if not finished.wait(_TIMEOUT_SECONDS):
            self.close()
            sender.join(timeout=1)
            raise PersonCrosscheckError("人员二检发送原帧超时；本次观察不可用")
        if errors:
            raise PersonCrosscheckError("人员二检发送原帧失败；本次观察不可用")

    def close(self):
        connection, process = self._connection, self._process
        self._connection = None
        self._process = None
        if connection is not None:
            connection.close()
        if process is not None and process.pid is not None:
            process.join(timeout=1)
            if process.is_alive():
                process.terminate()
                process.join(timeout=3)
            if process.is_alive():
                process.kill()
                process.join(timeout=3)
            if not process.is_alive():
                process.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
