/* Pure presentation helpers; no native capabilities or detector decisions. */
(function (root) {
  'use strict';
  const phases = Object.freeze({idle:'等待视频',loading:'读取视频',ready:'准备就绪',checking:'检查连接',running:'分析中',paused:'已暂停',stopping:'正在停止',finished:'分析完成',incomplete:'分析不完整',error:'需要处理'});
  function time(value) {
    if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) return '—';
    const n = Math.floor(value), h = Math.floor(n / 3600), m = Math.floor((n % 3600) / 60), s = n % 60;
    return (h ? String(h).padStart(2,'0') + ':' : '') + String(m).padStart(2,'0') + ':' + String(s).padStart(2,'0');
  }
  function label(phase) { return phases[phase] || '状态未知'; }
  function candidateTitle(item) {
    const value = item && typeof item.display_title === 'string' && item.display_title.trim() ? item.display_title : '待核：候选详情待核查';
    return value.startsWith('待核：') ? value : '待核：' + value;
  }
  function detailRows(rows) {
    return Array.isArray(rows) ? rows.slice(0,24).filter(row => row && typeof row.label === 'string' && typeof row.text === 'string' && row.text.trim()).map(row => ({label:row.label.slice(0,150),text:row.text.slice(0,1800)})) : [];
  }
  function candidateRenderKey(frameId, items, truncated) {
    return JSON.stringify([frameId, items, Boolean(truncated)]);
  }
  function buttons(state, connected) {
    const p = state && state.phase, video = Boolean(state && state.video), active = ['loading','checking','running','paused','stopping'].includes(p), occupied = active || Boolean(state && state.export_busy);
    return {choose:connected && !occupied, start:connected && !occupied && video && ['ready','finished','incomplete','error'].includes(p),
      pause:connected && ['running','paused'].includes(p), stop:connected && ['running','paused'].includes(p),
      settings:connected && !occupied, legacy:connected && !occupied,
      output:connected && Boolean(state && state.output_ready),
      export:connected && !active && Boolean(state && state.output_ready) && !state.export_busy};
  }
  const api = Object.freeze({time, label, buttons, candidateTitle, detailRows, candidateRenderKey});
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.VideoSafetyViewModel = api;
})(typeof window !== 'undefined' ? window : globalThis);
