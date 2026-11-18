(function () {
  'use strict';
  const $ = id => document.getElementById(id);
  const vm = window.VideoSafetyViewModel;
  let bridge = null;
  let current = null;
  let renderedCandidateKey = null;

  function node(tag, cls, text) {
    const el = document.createElement(tag);
    if (cls) el.className = cls;
    if (text !== undefined) el.textContent = text;
    return el;
  }
  function safeInt(n) { return Number.isSafeInteger(n) && n >= 0 ? n : 0; }
  function setDisabled(id, allowed) { $(id).disabled = !allowed; }
  function appendFields(parent, rows, cls) {
    vm.detailRows(rows).forEach(row => {
      const field = node('div',cls || 'candidate-field');
      field.append(node('span','candidate-label',row.label),node('p','candidate-evidence',row.text));
      parent.append(field);
    });
  }
  function renderFrames(rows, selected) {
    const list = $('frame-list');
    list.replaceChildren();
    if (!Array.isArray(rows) || !rows.length) { list.append(node('div','frame-empty','暂无分析帧')); return; }
    rows.forEach(row => {
      if (!Number.isSafeInteger(row.id) || row.id < 0) return;
      const button = node('button','frame-row' + (row.id === selected ? ' selected' : ''));
      button.type = 'button';
      button.append(node('span','frame-number',String(row.id + 1).padStart(2,'0')));
      const detail = node('span','frame-detail');
      detail.append(node('strong','',`帧 ${row.id + 1}`),node('small','',vm.time(row.timestamp)));
      button.append(detail);
      button.append(node('span',row.incomplete ? 'frame-badge warn' : 'frame-badge',row.incomplete ? '不完整' : `${safeInt(row.candidate_count)} 条`));
      button.addEventListener('click',() => { if (bridge) bridge.selectFrame(row.id); });
      list.append(button);
    });
  }
  function renderCandidates(items, truncated, hasFrame) {
    const list = $('candidate-list');
    list.replaceChildren();
    if (!Array.isArray(items) || !items.length) {
      const empty = node('div','candidate-empty');
      empty.append(node('span','','⌕'),node('strong','',hasFrame ? '该帧未输出候选' : '暂无候选详情'),node('p','',hasFrame ? '未输出候选不代表现场没有风险。' : '选择分析帧后，在此查看候选线索。'));
      list.append(empty); return;
    }
    items.forEach((item, i) => {
      if (!item || typeof item !== 'object') return;
      const card = node('article','candidate-card');
      const top = node('div','candidate-top');
      top.append(node('span','candidate-id',`线索 ${String(i+1).padStart(2,'0')}`),node('span','candidate-tag','待人工核查'));
      card.append(top,node('h3','',vm.candidateTitle(item)));
      card.append(node('p','candidate-source',typeof item.source_note === 'string' ? item.source_note : '来源未标注，需人工核查'));
      appendFields(card,item.evidence_notes,'candidate-warning');
      if (item.box_note) card.append(node('div','candidate-warning',item.box_note));
      else if (item.needs_box_relocalize) card.append(node('div','candidate-warning','位置需重新核查，原框可能偏移'));
      if (item.analysis_incomplete) card.append(node('div','candidate-warning','该候选关联的分析不完整'));
      const original = node('details','candidate-details');
      original.append(node('summary','','原始候选输出（未确认）'));
      appendFields(original,[
        {label:'原候选标题（待核）',text:item.title},
        {label:item.location_label || '位置（待核）',text:item.location},
        {label:item.description_label || '描述（待核）',text:item.description},
        {label:item.evidence_label || '观察（待核）',text:item.evidence}
      ]);
      card.append(original);
      if (vm.detailRows(item.review_diagnostics).length) {
        const diagnostics = node('details','candidate-details');
        diagnostics.append(node('summary','','复核与定位线索（不是人工结论）'));
        appendFields(diagnostics,item.review_diagnostics);
        card.append(diagnostics);
      }
      const context = node('details','candidate-details');
      context.append(node('summary','','场景与现场复核重点'));
      appendFields(context,item.context_advice);
      context.append(node('p','candidate-scope',typeof item.scope_note === 'string' ? item.scope_note : '逐帧候选不是最终事件，仍需人工核查。'));
      card.append(context);
      list.append(card);
    });
    if (truncated) list.append(node('div','candidate-warning','候选过多，仅展示前 200 条，请打开原始报告查看全部。'));
  }
  function render(state) {
    if (!state || state.schema !== 1) return;
    current = state;
    const controls = vm.buttons(state,Boolean(bridge));
    for (const [id,key] of [['choose-video','choose'],['start','start'],['pause','pause'],['stop','stop'],['settings','settings'],['legacy','legacy'],['open-output','output'],['export-evidence','export'],['export-notice','export']]) setDisabled(id,controls[key]);
    $('scene').disabled = !controls.choose;
    $('pause').textContent = state.phase === 'paused' ? '▶ 继续' : 'Ⅱ 暂停';
    $('stat-frames').textContent = String(safeInt(state.frame_count));
    $('stat-candidates').textContent = String(safeInt(state.candidate_count));
    $('stat-incomplete').textContent = String(safeInt(state.incomplete_count));
    $('sidebar-count').textContent = String(safeInt(state.candidate_count));
    $('frame-total').textContent = `${safeInt(state.frame_count)} 帧`;
    $('status-message').textContent = state.message || vm.label(state.phase);
    $('status-dot').className = 'status-dot ' + (['error','incomplete'].includes(state.phase) ? 'warning' : state.phase === 'running' ? 'running' : '');
    const progress = Math.max(0,Math.min(100,Number.isFinite(state.progress) ? state.progress : 0));
    $('progress-number').textContent = `${Math.round(progress)}%`;
    $('progress-bar').style.width = `${progress}%`;
    const video = state.video;
    $('video-name').textContent = video && video.name ? video.name : '选择本地视频';
    $('video-meta').textContent = video ? '视频已选择 · 点击可更换' : '支持 MP4、MOV、AVI 等格式';
    $('video-size').textContent = video ? `${video.width} × ${video.height}` : '—';
    $('video-duration').textContent = video ? vm.time(video.duration) : '—';
    $('video-fps').textContent = video && Number.isFinite(video.fps) ? `${video.fps.toFixed(1)} fps` : '—';
    const selected = (Array.isArray(state.frames) ? state.frames : []).find(x => x.id === state.selected);
    $('selected-time').textContent = selected ? vm.time(selected.timestamp) : '—';
    $('frame-label').textContent = selected ? `帧 ${selected.id+1}` : '未选择帧';
    const candidates = Array.isArray(state.selected_candidates) ? state.selected_candidates : [];
    $('selected-count').textContent = String(candidates.length);
    renderFrames(state.frames,state.selected);
    const key = vm.candidateRenderKey(selected ? selected.id : null, candidates, state.selected_candidates_truncated);
    // Progress is emitted many times during a run. Do not close details,
    // reset scroll or destroy focus when the selected candidate data is unchanged.
    if (key !== renderedCandidateKey) {
      renderCandidates(candidates,state.selected_candidates_truncated,Boolean(selected));
      renderedCandidateKey = key;
    }
  }
  function setPreview(value) {
    const valid = typeof value === 'string' && value.startsWith('data:image/jpeg;base64,') && value.length < 4_100_000;
    $('preview-image').hidden = !valid;
    $('preview-empty').hidden = valid;
    if (valid) $('preview-image').src = value;
    else $('preview-image').removeAttribute('src');
  }
  for (const [id,method] of [['choose-video','chooseVideo'],['pause','togglePause'],['stop','stopAnalysis'],['settings','openSettings'],['legacy','openLegacy'],['open-output','openOutput'],['export-evidence','exportEvidence'],['export-notice','exportNotice']]) {
    $(id).addEventListener('click',() => { if (bridge && typeof bridge[method] === 'function') bridge[method](); });
  }
  $('start').addEventListener('click',() => { if (bridge) bridge.startAnalysis($('scene').value); });
  document.querySelectorAll('[data-scroll]').forEach(button => button.addEventListener('click',() => {
    const target = $(button.dataset.scroll); if (target) target.scrollIntoView({behavior:'smooth',block:'start'});
  }));
  if (typeof QWebChannel === 'undefined' || typeof qt === 'undefined' || !qt.webChannelTransport) {
    $('bridge-status').textContent = '未连接桌面服务';
    return;
  }
  new QWebChannel(qt.webChannelTransport, channel => {
    bridge = channel.objects.bridge;
    if (!bridge) { $('bridge-status').textContent = '桌面服务不可用'; return; }
    $('bridge-status').replaceChildren(node('i'),document.createTextNode('桌面服务已连接'));
    bridge.stateChanged.connect(value => { try { render(JSON.parse(value)); } catch (_) { $('status-message').textContent = '界面状态读取失败'; } });
    bridge.previewChanged.connect(setPreview);
    bridge.ready();
  });
})();
