# -*- coding: utf-8 -*-
"""AI 复核「主队列／已移出主队列」口径的唯一来源（未确认事件不计入违章）。

QA 二轮发现对外文书口径漂移：report.txt / scene_summary 已按「主队列（kept）」
统计，但通知书与证据包仍把全部事件当确认违章（34 项 vs 实际 29 项）。把判定
与拆分集中到本模块，报告文本、场景汇总、通知书、证据包一律引用这里，避免各写
一份 ``review_disposition`` 元组后再次漂移。

口径约定：
- ``review_disposition in ("rejected", "box_relocalize", "dropped")`` 视为
  已移出主待复核队列，未确认，不计入违章统计；
- ``low_priority_review`` **不在**此列 —— 它 ``needs_review`` 仍为 true，
  继续留在主队列；
- 无任何处置（AI 复核开关关闭 / 旧版报告）时 ``moved`` 为空、``kept`` 与输入
  逐项一致，调用方输出逐字节不变；
- ``split_review_queue`` 只做筛选，不修改入参、不删除 ``report["events"]``；
  来源保护函数则显式原位标记待复核，不声称单条事件匹配金标。
"""

import hashlib
import os
import re
import stat


# 2026-11-12 QA 对当时六个本地来源文件的内容盘点。该清单只识别这些字节，
# 并不证明 2026-11-09 的历史报告由相同字节生成，更不关联任何单条事件。
REFERENCE_SOURCE_SHA256 = {
    '30eb2d4b34f2a243ecd0042f7abaeda6270479e3e16b462cc216bdaaae1cc804': '2e9f642707af3ea298756153bfbee935.mp4',
    '644066ccd393509b036f071768e1b1a925c626ff737feb3063497a308546a635': '7da29f116484e926c9f50aa0e1f5837c.mp4',
    '2b28f1e2aa7e592c1cfc9df10246e1ca34e93891cd8383e479086f47c0848250': '8aecce6836f83c1688a240a2058bcdd5.mp4',
    '2cd594f44083f20fcf99282717f2859d93976e95fd75ff98af4fdd1d18fc9240': 'a6f8c567a3d8da3dce5dad6ea94510ac.mp4',
    '88aaa16ed2c86eba70b6d014e31fdcbd7977ca20231f3dda2e9bd9f7e112df38': 'c46c885724b0305395c727f94c1b710e.mp4',
    'dd898ca97c97bfea997de87f384a4824ae1350affe19c5f7ebd33902c03bd124': 'fbffe054630b940cc2625a7fdf091e10.mp4',
}
_REFERENCE_SOURCE_MANIFEST_SHA256 = '00f8d32cd3d0935442cdff72a935ac41c541e376ed2910ab3f7fc20b3cbf4501'
_VIDEO_EXTENSIONS = frozenset(('.mp4', '.mov', '.m4v', '.avi', '.mkv', '.webm', '.wmv', '.flv',
                               '.mpeg', '.mpg', '.ts'))
_SOURCE_NOTE = ('来源视频级保守留队列，不表示事件匹配冻结样本；'
                '2026-11-12 内容摘要不能证明 2026-11-09 历史视频字节；'
                '摘要不同也不能排除同一场景的重编码或裁剪变体')


def _source_fingerprint(st):
    """不保存路径；用于检查处理前后的文件元数据是否一致。"""
    return {'device': st.st_dev, 'inode': st.st_ino,
            'size': st.st_size, 'mtime_ns': st.st_mtime_ns}


def _source_unknown(reason):
    return {'hold': True, 'status': 'source_unverified', 'reason': reason,
            'sha256': None, 'file_fingerprint': None, 'note': _SOURCE_NOTE}


def _valid_reference_sources():
    """来源资产无效时不能把任何视频错误当作已验证的非冻结来源。"""
    if not isinstance(REFERENCE_SOURCE_SHA256, dict):
        return False
    values = list(REFERENCE_SOURCE_SHA256.values())
    valid = (len(REFERENCE_SOURCE_SHA256) == 6
             and all(isinstance(key, str) and re.fullmatch(r'[0-9a-f]{64}', key)
                     for key in REFERENCE_SOURCE_SHA256)
             and all(isinstance(name, str) and re.fullmatch(r'[0-9a-f]{32}\.mp4', name)
                     for name in values) and len(set(values)) == 6)
    if not valid:
        return False
    lines = [key + ':' + REFERENCE_SOURCE_SHA256[key] for key in sorted(REFERENCE_SOURCE_SHA256)]
    return hashlib.sha256('\n'.join(lines).encode('ascii')).hexdigest() == _REFERENCE_SOURCE_MANIFEST_SHA256


def _verified_source_identity(identity):
    if not isinstance(identity, dict) or not _valid_reference_sources():
        return False
    digest, fingerprint = identity.get('sha256'), identity.get('file_fingerprint')
    if (not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest)
            or not isinstance(fingerprint, dict)
            or set(fingerprint) != {'device', 'inode', 'size', 'mtime_ns'}
            or any(type(value) is not int for value in fingerprint.values())):
        return False
    status, hold = identity.get('status'), identity.get('hold')
    if status == 'reference_digest_verified':
        return hold is True and digest in REFERENCE_SOURCE_SHA256
    if status == 'reference_filename_digest_conflict':
        return hold is True and digest not in REFERENCE_SOURCE_SHA256
    if status == 'nonreference_digest_verified':
        return hold is False and digest not in REFERENCE_SOURCE_SHA256
    return False


def classify_source_video(video_path, *, should_continue=None):
    """按来源文件内容分类；无法确认时保守持有，不读密钥或非视频文件。

    调用者应在单次分析开始时只计算一次，再于最终输出前调用
    :func:`recheck_source_video`。该结果仅是来源范围判定，不是金标事件匹配。
    """
    if not _valid_reference_sources():
        return _source_unknown('reference_manifest_invalid')
    try:
        path = os.fspath(video_path)
    except TypeError:
        return _source_unknown('invalid_path')
    if (not isinstance(path, str) or not path or not os.path.isabs(path)
            or path.startswith(('\\\\', '//'))):
        return _source_unknown('nonlocal_or_relative_path')
    if os.path.splitext(path)[1].lower() not in _VIDEO_EXTENSIONS:
        return _source_unknown('unsupported_extension')
    try:
        if os.path.islink(path):
            return _source_unknown('source_is_link')
        before_open = os.stat(path)
        if not stat.S_ISREG(before_open.st_mode):
            return _source_unknown('source_not_regular_file')
        with open(path, 'rb') as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                return _source_unknown('source_not_regular_file')
            digest = hashlib.sha256()
            while True:
                if should_continue is not None:
                    try:
                        if not should_continue():
                            return _source_unknown('source_hash_cancelled')
                    except Exception:
                        return _source_unknown('source_cancel_check_failed')
                block = stream.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
            after = os.fstat(stream.fileno())
        current = os.stat(path)
        stamp = _source_fingerprint(before)
        if (stamp != _source_fingerprint(before_open) or stamp != _source_fingerprint(after)
                or stamp != _source_fingerprint(current)):
            return _source_unknown('source_changed_during_hash')
    except (OSError, TypeError, ValueError):
        return _source_unknown('source_unreadable')
    value = digest.hexdigest()
    name = os.path.basename(path).casefold()
    expected_name = REFERENCE_SOURCE_SHA256.get(value)
    if expected_name is not None:
        status, hold = 'reference_digest_verified', True
    elif name in set(REFERENCE_SOURCE_SHA256.values()):
        # 同名替换不能当成已验证的其他视频；留队列并明确冲突。
        status, hold = 'reference_filename_digest_conflict', True
    else:
        status, hold = 'nonreference_digest_verified', False
    return {'hold': hold, 'status': status, 'reason': '', 'sha256': value,
            'file_fingerprint': stamp, 'note': _SOURCE_NOTE}


def recheck_source_video(video_path, identity, *, should_continue=None):
    """写报告前重新计算摘要；缺失、替换或变化时回退保守持有。

    前后摘要一致不能证明整个处理期间没有短暂替换；如需防并发恶意改写，
    需要只读快照或源文件锁。本函数本身不生成任何事件或视频分析结果。
    """
    if not _verified_source_identity(identity):
        return _source_unknown('initial_source_unverified')
    fresh = classify_source_video(video_path, should_continue=should_continue)
    if not _verified_source_identity(fresh):
        return _source_unknown(fresh.get('reason') or 'source_unreadable_after_hash')
    if any(fresh.get(key) != identity.get(key) for key in ('status', 'sha256', 'file_fingerprint')):
        return _source_unknown('source_changed_after_hash')
    return fresh


def source_video_review_guard(event):
    """来源级待复核独立于单事件金标豁免与几何护栏。"""
    return isinstance(event, dict) and event.get('source_video_review_hold') is True


def apply_source_video_review_hold(event, identity=None):
    """仅修改事件处置与留痕，不删除事件，不断言匹配冻结样本。"""
    if not isinstance(event, dict):
        return False
    if identity is not None:
        valid_other = (_verified_source_identity(identity)
                       and identity.get('status') == 'nonreference_digest_verified')
        if valid_other and not source_video_review_guard(event):
            return False
        if not valid_other:
            valid_unknown = (isinstance(identity, dict) and identity.get('hold') is True
                             and identity.get('status') == 'source_unverified')
            safe_identity = (identity if _verified_source_identity(identity) or valid_unknown
                             else _source_unknown('invalid_identity'))
            event['source_video_review_hold'] = True
            event['source_video_review_status'] = safe_identity['status']
            event['source_video_review_reason'] = _SOURCE_NOTE
    if not source_video_review_guard(event):
        return False
    previous = str(event.get('review_disposition') or '')
    if previous in (*UNCONFIRMED_DISPOSITIONS, 'low_priority_confirm'):
        event.setdefault('source_video_blocked_disposition', previous)
        if previous == 'box_relocalize':
            event['needs_box_relocalize'] = True
        event['review_disposition'] = 'low_priority_review'
    if event.get('needs_review') is False:
        event.setdefault('source_video_prior_needs_review', False)
    event['needs_review'] = True
    if event.get('ai_review_confirmed') is True:
        event['source_video_prior_ai_review_confirmed'] = True
        event.pop('ai_review_confirmed', None)
    return True

UNCONFIRMED_DISPOSITIONS = ("rejected", "box_relocalize", "dropped")


def enforce_incomplete_review_guard(event) -> bool:
    """部分失败候选必须留人工复核，不伪造已完成的 AI 处置。"""
    if not isinstance(event, dict) or event.get("analysis_incomplete") is not True:
        return False
    previous = str(event.get("review_disposition") or "")
    if previous in UNCONFIRMED_DISPOSITIONS:
        event["analysis_incomplete_blocked_disposition"] = previous
    # 显式处置使 AI 复核入口跳过不完整候选，避免 rejected 审计与最终队列矛盾。
    event["review_disposition"] = "low_priority_review"
    event["needs_review"] = True
    reason = event.get("review_reason")
    if not isinstance(reason, str) or not reason:
        event["review_reason"] = "逐帧分析不完整，候选仅供人工核查"
    elif "分析不完整" not in reason:
        event["review_reason"] = reason + "；逐帧分析不完整，候选仅供人工核查"
    return True


def hold_text_overlay_check(event, status):
    """文本重叠或检查失败只形成待核保护，不删除候选、不改变原框。"""
    event['text_overlay_check_status'] = status
    event['analysis_incomplete'] = True
    previous = str(event.get('review_disposition') or '')
    if previous and previous != 'low_priority_review':
        event.setdefault('text_overlay_prior_disposition', previous)
    if event.pop('ai_review_confirmed', None) is True:
        event['text_overlay_prior_ai_review_confirmed'] = True
    if 'ai_review_verdict' in event:
        event.setdefault('text_overlay_prior_ai_review_verdict', event.pop('ai_review_verdict'))
    reason = str(event.get('review_reason') or '')
    note = ('候选框与疑似文字区域重叠，目标位置需重新核验，分析不完整'
            if status == 'suspected_text_overlap' else '文字重叠检查未完成，分析不完整')
    if note not in reason:
        event['review_reason'] = (reason + '；' if reason else '') + note
    if status == 'suspected_text_overlap':
        event['needs_box_relocalize'] = True
    return enforce_incomplete_review_guard(event)


def is_unconfirmed(event) -> bool:
    """该事件是否已被 AI 复核移出主队列（未确认）。"""
    disposition = str((event or {}).get("review_disposition") or "")
    return disposition in UNCONFIRMED_DISPOSITIONS


def split_review_queue(events):
    """按 AI 复核处置拆分报告事件：返回 ``(主队列事件, 已移出主队列事件)``。

    顺序稳定（各自保持入参相对顺序），不修改输入列表与事件本体。
    """
    kept, moved = [], []
    for event in events or []:
        if is_unconfirmed(event):
            moved.append(event)
        else:
            kept.append(event)
    return kept, moved
