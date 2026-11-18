"""同一可核验原帧上执行本地实体检测，供轨迹和机械候选共用。

源帧号、精确时间和 JPEG 摘要缺失、冲突或变化时不能借近邻帧；缓存只证明
像素来源一致，不证明检测主体是真人/机械，亦不证明来源记录本身不可伪造。
"""
import copy
import math
from collections import Counter

from core.event_image_sync import verified_exact_original
from core.frame_lookup import normalize_timestamp


class VerifiedEntityCache(dict):
    """保留兼容时间键及独立来源审计；取用时再次核验当前原图字节。"""

    def __init__(self, output_dir, rows):
        super().__init__()
        self.output_dir = output_dir
        self.rows = [dict(row) for row in rows if isinstance(row, dict)]
        self.by_key = {}
        self.audit = {'input_rows': len(rows), 'verified': 0, 'skipped': 0, 'reasons': {}}

    def skip(self, reason):
        self.audit['skipped'] += 1
        reasons = self.audit['reasons']
        reasons[reason] = reasons.get(reason, 0) + 1


def _identity(row):
    if not isinstance(row, dict):
        return None
    index, timestamp = row.get('frame_index'), normalize_timestamp(row.get('timestamp'))
    if type(index) is not int or index < 0 or timestamp is None:
        return None
    return index, timestamp


def _original(cache, row):
    identity = _identity(row)
    if identity is None:
        return None, None, 'invalid_frame_identity'
    index, timestamp = identity
    return verified_exact_original({'representative_frame': index, 'representative_timestamp': timestamp},
                                   cache.rows, cache.output_dir)


def build_entity_cache(out_dir, frame_results, det):
    """对每个唯一时间键只读取精确原帧；原图或检测器失败不会伪装成空检测。"""
    rows = frame_results if isinstance(frame_results, (list, tuple)) else []
    cache = VerifiedEntityCache(out_dir, rows)
    keys = [round(identity[1], 1) for row in rows if (identity := _identity(row)) is not None]
    counts = Counter(keys)
    for row in rows:
        identity = _identity(row)
        if identity is None:
            cache.skip('invalid_frame_identity')
            continue
        index, timestamp = identity
        key = round(timestamp, 1)
        if counts[key] != 1:
            cache.skip('rounded_timestamp_collision')
            continue
        record, image, status = _original(cache, row)
        if status != 'ok':
            cache.skip(status)
            continue
        try:
            prior = det.build_scene_prior(image)
            entities = prior.get('entities') if isinstance(prior, dict) else None
            if not isinstance(entities, list) or any(not isinstance(item, dict) for item in entities):
                raise ValueError('invalid_detector_output')
        except Exception:
            cache.skip('detector_failed')
            continue
        _, image_after, after_status = _original(cache, row)
        if after_status != 'ok' or image_after is None:
            cache.skip('frame_changed_during_detection')
            continue
        cache[key] = copy.deepcopy(entities)
        cache.by_key[key] = {'frame_index': index, 'timestamp': timestamp,
                             'sha256': record['original_frame_sha256'],
                             'original_path': record.get('original_path'),
                             'height': int(image.shape[0]), 'width': int(image.shape[1])}
        cache.audit['verified'] += 1
    return cache


def get_for_row(cache, row):
    """返回 ``(实体列表或 None, 是否核验)``；普通旧字典仅为未核验线索。"""
    identity = _identity(row)
    if identity is None or cache is None:
        return None, False
    index, timestamp = identity
    key = round(timestamp, 1)
    if not isinstance(cache, VerifiedEntityCache):
        if isinstance(cache, dict):
            return cache.get(key), False
        return None, False
    info = cache.by_key.get(key)
    if (not isinstance(info, dict) or info.get('frame_index') != index
            or not math.isclose(info.get('timestamp', -1), timestamp, abs_tol=1e-6, rel_tol=0)
            or row.get('original_frame_sha256') != info.get('sha256')
            or row.get('original_path') != info.get('original_path')):
        return None, False
    _, image, status = _original(cache, row)
    if status != 'ok' or image is None:
        return None, False
    entities = dict.get(cache, key)
    return copy.deepcopy(entities), isinstance(entities, list)


def get(cache, ts):
    """兼容旧时间查询；带溯源缓存按记录重新核验，非法时间不映射到零。"""
    timestamp = normalize_timestamp(ts)
    if not cache or timestamp is None:
        return None
    key = round(timestamp, 1)
    if isinstance(cache, VerifiedEntityCache):
        info = cache.by_key.get(key)
        if not isinstance(info, dict) or not math.isclose(info['timestamp'], timestamp, abs_tol=1e-6, rel_tol=0):
            return None
        row = next((item for item in cache.rows if item.get('frame_index') == info['frame_index']), None)
        entities, verified = get_for_row(cache, row)
        return entities if verified else None
    return cache.get(key) if isinstance(cache, dict) else None
