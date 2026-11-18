"""Presentation-neutral session state; it never changes detector or review decisions."""
import math
import os
from application.candidate_presenter import present_candidate
from application.presentation_text import clean_text
from core.report_status import frame_analysis_incomplete


ACTIVE = frozenset({'loading', 'checking', 'running', 'paused', 'stopping'})
STARTABLE = frozenset({'ready', 'finished', 'incomplete', 'error'})


def _number(value, default=0.0):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    value = float(value)
    return value if math.isfinite(value) else default


def _frame(index, timestamp, annotated, original, result):
    result = result if isinstance(result, dict) else {}
    raw = result.get('violations')
    raw = raw if isinstance(raw, list) else []
    scene = next((value for key in ('scene_name', 'scene')
                  if isinstance(value := result.get(key), str) and value.strip()), None)
    candidates = [v for v in (present_candidate(x, frame_scene=scene) for x in raw[:200]) if v is not None]
    count = sum(isinstance(item, dict) for item in raw)
    return {'id': index, 'timestamp': max(0.0, _number(timestamp)),
            'incomplete': frame_analysis_incomplete(result),
            'candidates': candidates, 'candidate_count': count,
            'candidates_truncated': count > len(candidates),
            # Paths remain in Python and are never included in the published snapshot.
            '_annotated': annotated if isinstance(annotated, str) else '',
            '_original': original if isinstance(original, str) else ''}


class AnalysisSession:
    """State machine for one selected file and one analysis at a time."""

    def __init__(self):
        self.phase = 'idle'
        self.video_path = ''
        self.video = None
        self.progress = 0
        self.message = '选择视频后开始分析'
        self.frames = {}
        self.selected = None
        self.output_dir = ''

    def begin_loading(self):
        if self.phase in ACTIVE:
            return False
        self.phase, self.message = 'loading', '正在读取视频信息'
        return True

    def video_loaded(self, path, info):
        if self.phase != 'loading' or not isinstance(path, str) or not path or not isinstance(info, dict):
            return False
        width, height = int(_number(info.get('width'))), int(_number(info.get('height')))
        if width <= 0 or height <= 0:
            return self.loading_failed()
        self.video_path = path
        self.video = {'name': clean_text(os.path.basename(path), 180), 'width': width, 'height': height,
                      'fps': max(0.0, _number(info.get('fps'))),
                      'duration': max(0.0, _number(info.get('duration')))}
        self.frames, self.selected, self.output_dir = {}, None, ''
        self.progress, self.phase, self.message = 0, 'ready', '视频已就绪，可开始分析'
        return True

    def loading_failed(self):
        if self.phase != 'loading':
            return False
        self.phase = 'ready' if self.video else 'idle'
        self.message = '无法读取所选视频，请检查文件或编码'
        return False

    def begin_check(self):
        if self.phase not in STARTABLE or not self.video_path:
            return False
        self.phase, self.message = 'checking', '正在检查分析服务连接'
        return True

    def connection_failed(self):
        if self.phase != 'checking':
            return False
        self.phase, self.message = 'error', '连接检查失败，请检查设置；未启动视频分析'
        return True

    def begin_analysis(self):
        if self.phase != 'checking':
            return False
        self.frames, self.selected, self.output_dir = {}, None, ''
        self.phase, self.progress, self.message = 'running', 0, '分析已启动'
        return True

    def set_paused(self, paused):
        if paused and self.phase == 'running':
            self.phase, self.message = 'paused', '已暂停，进行中的请求可能仍需返回'
            return True
        if not paused and self.phase == 'paused':
            self.phase, self.message = 'running', '已继续分析'
            return True
        return False

    def stop(self):
        if self.phase not in ('running', 'paused'):
            return False
        self.phase, self.message = 'stopping', '正在停止，等待进行中的任务结束'
        return True

    def set_progress(self, value, message):
        if self.phase not in ('running', 'paused', 'stopping'):
            return
        self.progress = max(0, min(100, int(_number(value))))
        if self.phase == 'running':
            self.message = clean_text(message, 180) or '分析中'

    def add_frame(self, index, timestamp, annotated, original, result):
        if self.phase not in ('running', 'paused', 'stopping') or type(index) is not int or not 0 <= index <= 1_000_000:
            return False
        self.frames[index] = _frame(index, timestamp, annotated, original, result)
        if self.selected is None:
            self.selected = index
        return True

    def finish(self, rows, output_dir, *, complete):
        if self.phase not in ('running', 'paused', 'stopping'):
            return False
        complete = complete is True and self.phase != 'stopping'
        if isinstance(rows, list) and rows:
            rebuilt = {}
            for index, row in enumerate(rows):
                if isinstance(row, dict):
                    rebuilt[index] = _frame(index, row.get('timestamp'), row.get('annotated_path'),
                                            row.get('original_path'), row.get('analysis'))
            # A stop/error callback may have no final rows. Keep already emitted
            # provisional frames visible rather than silently wiping evidence.
            if rebuilt:
                self.frames = rebuilt
        self.selected = min(self.frames) if self.frames else None
        self.output_dir = output_dir if isinstance(output_dir, str) else ''
        self.phase = 'finished' if complete else 'incomplete'
        self.progress = 100 if complete else self.progress
        self.message = ('分析完成；所有候选仍需人工核查' if complete else
                        '分析不完整或已停止；不能据此判定现场安全')
        return True

    def failed(self):
        if self.phase in ('running', 'paused', 'stopping'):
            self.phase, self.message = 'error', '分析失败，已保留现有结果；请检查配置和日志'
            return True
        return False

    def select_frame(self, index):
        if type(index) is not int or index not in self.frames:
            return False
        self.selected = index
        return True

    def snapshot(self):
        ordered = [self.frames[i] for i in sorted(self.frames)]
        selected = self.frames.get(self.selected)
        return {'schema': 1, 'phase': self.phase, 'video': self.video,
                'progress': self.progress, 'message': clean_text(self.message, 180),
                'frame_count': len(ordered),
                'candidate_count': sum(x['candidate_count'] for x in ordered),
                'incomplete_count': sum(x['incomplete'] for x in ordered),
                'output_ready': bool(self.output_dir), 'selected': self.selected,
                'frames': [{k: row[k] for k in ('id', 'timestamp', 'incomplete', 'candidate_count')}
                           for row in ordered],
                'selected_candidates': list(selected['candidates']) if selected else [],
                'selected_candidates_truncated': selected['candidates_truncated'] if selected else False}
