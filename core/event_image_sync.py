"""把最终事件框画在其确切代表帧上，并保留精修前证据图。

同步只保证图片与报告中的坐标一致，不代表框住了正确目标。
"""
import hashlib
import math
import os
import shutil
from collections import Counter

import cv2
import numpy as np

from core.annotation_renderer import (
    _resolve_bbox,
    annotate_image,
    render_detailed_event_card,
    save_annotated_image,
)
from core.frame_lookup import normalize_timestamp


def _same_time(a, b):
    a, b = normalize_timestamp(a), normalize_timestamp(b)
    return a is not None and b is not None and math.isclose(a, b, abs_tol=1e-6, rel_tol=0)


def _find_exact_frame(event, results):
    """帧号与时间均有效且唯一；无帧号的规则事件要求唯一精确时间。"""
    if not isinstance(event, dict):
        return None
    frame = event.get('representative_frame')
    timestamp = event.get('representative_timestamp')
    if normalize_timestamp(timestamp) is None:
        return None
    rows = [r for r in results or [] if isinstance(r, dict)]
    if frame is not None:
        if type(frame) is not int or frame < 0:
            return None
        matches = [r for r in rows if r.get('frame_index') == frame]
        if (len(matches) != 1 or type(matches[0].get('frame_index')) is not int
                or not _same_time(matches[0].get('timestamp'), timestamp)):
            return None
    else:
        matches = [r for r in rows if _same_time(r.get('timestamp'), timestamp)]
    if len(matches) != 1 or type(matches[0].get('frame_index')) is not int or matches[0]['frame_index'] < 0:
        return None
    return matches[0]


def verified_exact_original(event, results, output_dir):
    """返回 (对应帧记录, 原图, 状态)；仅 ``ok`` 表示位置与落盘字节均可核验。

    带帧号的事件需唯一帧号及精确时间；无帧号的规则事件仅用唯一精确时间。
    帧摘要缺失或不符均不得把文件当作最终证据，且不读取视频或运行检测。
    """
    row = _find_exact_frame(event, results)
    if row is None:
        return None, None, 'no_exact_frame'
    if row.get('annotation_frame_status') == 'unverified_frame_source':
        return row, None, 'unverified_frame_source'
    digest = row.get('original_frame_sha256')
    if (not isinstance(digest, str) or len(digest) != 64
            or any(c not in '0123456789abcdef' for c in digest)):
        return row, None, 'unverified_frame_source'
    source = row.get('original_path')
    if not source:
        annotated = row.get('annotated_path') or ''
        source = os.path.join(output_dir, 'original', os.path.basename(annotated))
    if not isinstance(source, str) or not source:
        return row, None, 'unsafe_source'
    try:
        output_root = os.path.realpath(output_dir)
        root = os.path.realpath(os.path.join(output_dir, 'original'))
        if (os.path.commonpath((output_root, root)) != output_root or os.path.islink(source)
                or os.path.commonpath((root, os.path.realpath(source))) != root):
            return row, None, 'unsafe_source'
        with open(source, 'rb') as stream:
            data = stream.read()
        if hashlib.sha256(data).hexdigest() != digest:
            return row, None, 'unverified_frame_source'
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return row, None, 'unreadable_frame'
        return row, image, 'ok'
    except (OSError, ValueError, TypeError, cv2.error, MemoryError):
        return row, None, 'unreadable_frame'


def _sha256(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _invalidate_synced_image(event, status):
    """移除当前证据引用但不删除文件，旧引用仅供审计，不得冒充最终证据。"""
    prior = {}
    for key in ('event_image', 'event_image_pre_refine', 'event_image_bbox', 'event_image_sha256',
                'event_image_source_frame_sha256'):
        if key in event:
            prior[key] = event.pop(key)
    if prior:
        prior['reference_scope'] = 'historic_path_content_not_guaranteed'
        history = event.get('event_image_unverified_history')
        if not isinstance(history, list):
            history = []
            event['event_image_unverified_history'] = history
        history.append(dict(prior))
        event['event_image_prior_unverified'] = prior
    event['event_image_status'] = status


def sync_event_images(events, results, output_dir):
    """同步最终报告事件图；失败时留旧图且明确标为不可用于最终证据。"""
    counts = Counter(str(e.get('event_id')) for e in events if isinstance(e, dict))
    stats = Counter()
    events_dir = os.path.join(output_dir, 'events')
    prior_dir = os.path.join(output_dir, 'events_pre_refine')
    for event in events:
        if not isinstance(event, dict):
            continue
        if any(key in event for key in ('event_image', 'event_image_pre_refine',
                                        'event_image_bbox', 'event_image_sha256',
                                        'event_image_source_frame_sha256')):
            _invalidate_synced_image(event, 'resync_pending')
        event['event_image_status'] = 'unavailable'
        event.pop('event_image', None)
        event.pop('event_image_pre_refine', None)
        event.pop('event_image_bbox', None)
        event.pop('event_image_sha256', None)
        event.pop('event_image_source_frame_sha256', None)
        eid = event.get('event_id')
        if isinstance(eid, bool) or not isinstance(eid, int) or eid < 0 or counts[str(eid)] != 1:
            stats['ambiguous_id'] += 1
            continue
        frame, image, frame_status = verified_exact_original(event, results, output_dir)
        if frame_status != 'ok':
            if frame_status == 'unverified_frame_source':
                # 来源摘要缺失或不一致；原文件和旧事件图均保留。
                event['event_image_status'] = 'unverified_frame_source'
            stats[frame_status] += 1
            continue
        if image is None:
            _invalidate_synced_image(event, 'unverified_frame_source')
            stats['unverified_frame_source'] += 1
            continue
        try:
            if event.get('text_overlay_check_status') == 'suspected_text_overlap':
                # 旧图仅保留审计引用，不把字幕框继续发布成目标定位证据。
                event['event_image_status'] = 'text_overlap_unlocalized'
                stats['text_overlap_unlocalized'] += 1
                continue
            if event.get('scene_level') and not event.get('bbox'):
                event['event_image_status'] = 'scene_level_no_box'
                stats['scene_level_no_box'] += 1
                continue
            bx, by, bw, bh, valid = _resolve_bbox(event.get('bbox'), image.shape[1], image.shape[0])
            if not valid:
                event['event_image_status'] = 'no_valid_bbox'
                stats['no_valid_bbox'] += 1
                continue
            filename = f'event_{eid:02d}.jpg'
            target = os.path.join(events_dir, filename)
            prior = os.path.join(prior_dir, filename)
            os.makedirs(events_dir, exist_ok=True)
            if os.path.isfile(target):
                os.makedirs(prior_dir, exist_ok=True)
                target_hash = _sha256(target)
                if os.path.exists(prior) and _sha256(prior) != target_hash:
                    prior = os.path.join(prior_dir, f'event_{eid:02d}.{target_hash}.jpg')
                if not os.path.exists(prior):
                    shutil.copy2(target, prior)
                if _sha256(prior) != target_hash:
                    stats['backup_failed'] += 1
                    continue
                event['event_image_pre_refine'] = 'events_pre_refine/' + os.path.basename(prior)
            rendered = annotate_image(image, {'violations': [event]})
            temporary = target + '.final.tmp.jpg'
            if rendered is None or not save_annotated_image(rendered, temporary):
                stats['render_failed'] += 1
                continue
            image_hash = _sha256(temporary)
            os.replace(temporary, target)
            if _sha256(target) != image_hash:
                stats['write_or_read_failed'] += 1
                continue
            event['event_image'] = 'events/' + filename
            event['event_image_status'] = 'synchronized'
            event['event_image_bbox'] = {'x': bx, 'y': by, 'w': bw, 'h': bh}
            event['event_image_sha256'] = image_hash
            event['event_image_source_frame_sha256'] = frame['original_frame_sha256']
            stats['synchronized'] += 1
        except (OSError, ValueError, TypeError, cv2.error):
            # 原因不包含外部路径或异常正文；老证据保留，不能冒充最终框证据。
            stats['write_or_read_failed'] += 1
    return dict(stats)


def create_event_detail_cards(events, results, output_dir):
    """另存图外详情卡；不改原图、原框、事件内容或自动复核状态。"""
    counts = Counter(str(e.get('event_id')) for e in events if isinstance(e, dict))
    stats = Counter()
    for event in events:
        if not isinstance(event, dict):
            continue
        event['detail_card_status'] = 'unavailable'
        event.pop('detail_card_path', None)
        event.pop('detail_card_meta', None)
        event.pop('detail_card_source', None)
        event.pop('detail_card_sha256', None)
        eid = event.get('event_id')
        if isinstance(eid, bool) or not isinstance(eid, int) or eid < 0 or counts[str(eid)] != 1:
            if event.get('event_image_status') == 'synchronized':
                _invalidate_synced_image(event, 'ambiguous_id')
            stats['ambiguous_id'] += 1
            continue
        state = event.get('event_image_status')
        if state not in ('synchronized', 'scene_level_no_box', 'no_valid_bbox'):
            if state == 'unverified_frame_source':
                event['detail_card_status'] = 'unverified_frame_source'
                stats['unverified_frame_source'] += 1
            else:
                stats['no_safe_source'] += 1
            continue
        frame, original_image, frame_status = verified_exact_original(event, results, output_dir)
        if frame_status != 'ok':
            # 阶段间原图变化时旧事件图仍留档，但不能继续标为已验证最终证据。
            _invalidate_synced_image(event, 'unverified_frame_source')
            if frame_status == 'unverified_frame_source':
                event['detail_card_status'] = 'unverified_frame_source'
            stats[frame_status] += 1
            continue
        card_event = dict(event)
        if state == 'synchronized':
            source = os.path.join(output_dir, 'events', f'event_{eid:02d}.jpg')
            source_kind = 'synchronized_event_image'
            image_hash = event.get('event_image_sha256')
            if (event.get('event_image_source_frame_sha256') != frame.get('original_frame_sha256')
                    or not isinstance(image_hash, str) or len(image_hash) != 64
                    or any(c not in '0123456789abcdef' for c in image_hash)):
                _invalidate_synced_image(event, 'unverified_event_image')
                event['detail_card_status'] = 'unverified_event_image'
                stats['unverified_event_image'] += 1
                continue
        else:
            source = frame.get('original_path') or os.path.join(output_dir, 'original',
                                                               os.path.basename(frame.get('annotated_path') or ''))
            source_kind = 'original_frame_without_box'
            card_event['bbox'] = None
        image_verified = False
        try:
            output_root = os.path.realpath(output_dir)
            source_root = os.path.realpath(os.path.join(
                output_dir, 'events' if source_kind == 'synchronized_event_image' else 'original'))
            if (os.path.commonpath((output_root, source_root)) != output_root
                    or os.path.islink(source)
                    or os.path.commonpath((source_root, os.path.realpath(source))) != source_root):
                if state == 'synchronized':
                    _invalidate_synced_image(event, 'unverified_event_image')
                    event['detail_card_status'] = 'unverified_event_image'
                stats['unsafe_source'] += 1
                continue
            if source_kind == 'synchronized_event_image':
                with open(source, 'rb') as stream:
                    data = stream.read()
                if hashlib.sha256(data).hexdigest() != image_hash:
                    _invalidate_synced_image(event, 'unverified_event_image')
                    event['detail_card_status'] = 'unverified_event_image'
                    stats['unverified_event_image'] += 1
                    continue
                image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            else:
                image = original_image
            if image is None:
                if state == 'synchronized':
                    _invalidate_synced_image(event, 'unverified_event_image')
                    event['detail_card_status'] = 'unverified_event_image'
                stats['unreadable_frame'] += 1
                continue
            image_verified = True
            card, meta = render_detailed_event_card(image, card_event)
            if card is None:
                event['detail_card_status'] = str(meta.get('status') or 'render_failed')
                stats['render_failed'] += 1
                continue
            ok, encoded = cv2.imencode('.png', card)
            if not ok:
                stats['encode_failed'] += 1
                continue
            folder = os.path.join(output_dir, 'events_detail')
            os.makedirs(folder, exist_ok=True)
            if os.path.commonpath((output_root, os.path.realpath(folder))) != output_root:
                stats['unsafe_destination'] += 1
                continue
            filename = f'event_{eid:02d}.png'
            destination = os.path.join(folder, filename)
            if os.path.islink(destination):
                stats['unsafe_destination'] += 1
                continue
            # 已存在时不覆盖；同内容可复用，不同内容保守拒绝，保留旧证据。
            if os.path.exists(destination):
                if _sha256(destination) != hashlib.sha256(encoded.tobytes()).hexdigest():
                    event['detail_card_status'] = 'existing_conflict'
                    stats['existing_conflict'] += 1
                    continue
            else:
                with open(destination, 'xb') as target:
                    encoded.tofile(target)
            event['detail_card_path'] = 'events_detail/' + filename
            event['detail_card_status'] = 'ready'
            event['detail_card_source'] = source_kind
            event['detail_card_meta'] = meta
            event['detail_card_sha256'] = hashlib.sha256(encoded.tobytes()).hexdigest()
            stats['ready'] += 1
        except (OSError, ValueError, TypeError, cv2.error, MemoryError):
            if state == 'synchronized' and not image_verified:
                _invalidate_synced_image(event, 'unverified_event_image')
                event['detail_card_status'] = 'unverified_event_image'
            stats['write_or_read_failed'] += 1
    return dict(stats)
