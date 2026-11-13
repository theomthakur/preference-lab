'use strict';
/* Preference Lab dashboard. Read-only: it only displays data.json. Every number shown is derived from that file, so the text cannot drift
   from the saved results. Elements that can hold keyboard focus are never replaced while they are focused: state lives in attributes. */

const MODELS = ['base', 'sft', 'dpo_clean', 'dpo_corrupt', 'grpo'];
const NAMES = { base: 'Base model', sft: 'SFT', dpo_clean: 'DPO, clean pairs', dpo_corrupt: 'DPO, flipped pairs', grpo: 'GRPO' };
const DESCRIPTIONS = { base: 'Before training', sft: 'Supervised examples', dpo_clean: 'Clean preferences', dpo_corrupt: '20% of labels flipped', grpo: 'Verifier reward' };
const VIEW_LABELS = { all_rows: 'All rows', seen_template: 'Seen wording', held_out_template: 'Held-out wording' };
const PAIR_LABELS = { sft_vs_base: ['SFT', 'base model'], dpo_clean_vs_sft: ['DPO (clean pairs)', 'SFT'], dpo_corrupt_vs_dpo_clean: ['DPO (flipped pairs)', 'DPO (clean pairs)'], grpo_vs_sft: ['GRPO', 'SFT'] };
const VIEW_TITLES = { overview: 'Overview', 'system-design': 'System design', presentation: 'Presentation' };
const ANCHORS = { explore: 'overview', results: 'overview' };
const PER_PAGE = 7;
const HELD_OUT_TEMPLATE = 3;

const $ = id => document.getElementById(id);
const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const pct = v => `${(v * 100).toFixed(2).replace(/\.?0+$/, '')}%`;
const pct1 = v => `${(v * 100).toFixed(1)}%`;
const signed1 = v => `${v < 0 ? '−' : '+'}${Math.abs(v * 100).toFixed(1)}`;
const reducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const scrollBehavior = () => (reducedMotion() ? 'auto' : 'smooth');
const isNarrow = () => window.matchMedia('(max-width: 900px)').matches;
const zeroUpper95 = n => 1 - Math.exp(Math.log(0.025) / n);   // exact two-sided 95% upper bound when 0 of n succeed

const state = { data: null, filtered: [], page: 0, selectedId: null, model: 'sft', view: 'all_rows', slide: 0 };

function announce(message) {
  const s = $('status'); s.textContent = '';
  window.setTimeout(() => { s.textContent = message; }, 40);
}
const wordingOf = c => (c.template === HELD_OUT_TEMPLATE ? 'held-out wording' : 'seen wording');
const metric = (model, view = 'all_rows') => state.data.metrics.models[model][view];
const groupsIn = view => new Set(state.data.cases.filter(c => view === 'all_rows' || (view === 'held_out_template') === (c.template === HELD_OUT_TEMPLATE)).map(c => c.group_id)).size;

/* ---------- results: findings, bars with intervals, paired table ---------- */
function readPair(p) {
  const [lo, hi] = p.paired_diff_ci95;
  if (lo > 0) return 'Higher';
  if (hi < 0) return 'Lower';
  return 'No clear difference';
}

function renderGlance() {
  const d = state.data, sft = metric('sft'), base = metric('base');
  const items = [];
  items.push(`Training helped. SFT scored ${pct(sft.point)} on the held-out test, against ${pct(base.point)} for the untrained model.`);
  const unclear = ['dpo_clean_vs_sft', 'dpo_corrupt_vs_dpo_clean', 'grpo_vs_sft'].filter(k => readPair(d.metrics.paired[k]) === 'No clear difference');
  items.push(unclear.length === 3
    ? 'DPO and GRPO were not clearly different from SFT: the paired intervals include zero. This experiment does not show that either improved on SFT.'
    : 'Some DPO or GRPO differences from SFT are clear; see the paired table for which ones.');
  const seen = metric('sft', 'seen_template').point, held = metric('sft', 'held_out_template').point;
  const half = held / seen >= 0.4 && held / seen <= 0.6 ? ', about half as high' : '';
  items.push(`New wording was harder. SFT scored ${pct(seen)} on wording it had seen and ${pct(held)} on wording it had not${half}.`);
  $('glance').innerHTML = items.map(t => `<li>${esc(t)}</li>`).join('');
}

function axisMax() {
  let max = 0;
  for (const m of MODELS) for (const v of Object.keys(VIEW_LABELS)) max = Math.max(max, metric(m, v).ci95[1]);
  return Math.ceil((max * 100) / 5) * 5 / 100;   // one fixed axis for all three views, so switching views is a fair comparison
}

function renderBars() {
  const view = state.view, scale = axisMax();
  const rows = MODELS.map(m => {
    const x = metric(m, view), k = Math.round(x.point * x.n_rows), [lo, hi] = x.ci95;
    const detail = k === 0 ? `0 of ${x.n_rows}. The interval is [0, 0], which shows no spread` : `${k} of ${x.n_rows} · 95% interval ${pct1(lo)} to ${pct1(hi)}`;
    const label = `${NAMES[m]}: ${pct(x.point)} strictly correct, ${k} of ${x.n_rows} ${VIEW_LABELS[view].toLowerCase()}. ${k === 0 ? 'The bootstrap interval is zero to zero, which shows no uncertainty.' : `95 percent interval ${pct1(lo)} to ${pct1(hi)}.`}`;
    return `<div class="bar-row" role="group" aria-label="${esc(label)}">
      <div class="bar-label"><b>${esc(NAMES[m])}</b><small>${esc(DESCRIPTIONS[m])}</small></div>
      <div class="bar-track" aria-hidden="true"><i class="bar-fill ${m}" data-fill="${x.point}"></i><i class="bar-ci" data-lo="${lo}" data-hi="${hi}"></i></div>
      <div class="bar-value"><b>${pct(x.point)}</b><small>${esc(detail)}</small></div>
    </div>`;
  }).join('');
  const ticks = []; for (let t = 0; t <= scale * 100 + 0.001; t += 5) ticks.push(t);
  const axis = `<div class="axis" aria-hidden="true"><span class="axis-spacer"></span><div class="axis-scale">${ticks.map(t => `<span class="tick" data-at="${t / 100}">${t}%</span>`).join('')}</div><span class="axis-spacer"></span></div>`;
  const box = $('bars');
  box.innerHTML = rows + axis;
  box.querySelectorAll('[data-fill]').forEach(e => { e.style.width = `${(Number(e.dataset.fill) / scale) * 100}%`; });
  box.querySelectorAll('.bar-ci').forEach(e => {
    const lo = Number(e.dataset.lo), hi = Number(e.dataset.hi);
    e.style.left = `${(lo / scale) * 100}%`; e.style.width = `${((hi - lo) / scale) * 100}%`;
  });
  box.querySelectorAll('.tick').forEach(e => { e.style.left = `${(Number(e.dataset.at) / scale) * 100}%`; });
  const total = metric('sft', view).n_rows;
  $('chart-sub').textContent = `${VIEW_LABELS[view]} · ${total} test rows from ${groupsIn(view)} groups · greedy decoding · one training seed per model`;
  const zeros = MODELS.filter(m => metric(m, view).point === 0);
  const note = $('zero-note');
  if (zeros.length) {
    const g = groupsIn(view), n = metric(zeros[0], view).n_rows;
    note.hidden = false;
    note.textContent = `Zero successes (${zeros.map(m => NAMES[m]).join(', ')}): a bootstrap interval is always [0, 0] when nothing is correct, so it does not show how uncertain the true rate is. As supplementary arithmetic, an exact 95% upper bound for 0 of ${g} groups is about ${(zeroUpper95(g) * 100).toFixed(2)}%, and for 0 of ${n} independent rows about ${(zeroUpper95(n) * 100).toFixed(2)}%. The groups figure is the more cautious one. These bounds are not part of the saved metrics.`;
  } else { note.hidden = true; note.textContent = ''; }
}

function renderPaired() {
  const rows = Object.entries(PAIR_LABELS).map(([key, [a, b]]) => {
    const p = state.data.metrics.paired[key], [lo, hi] = p.paired_diff_ci95;
    return `<tr><th scope="row">${esc(a)} minus ${esc(b)}</th><td data-label="Difference">${signed1(p.paired_diff_point)} pts</td><td data-label="95% interval">${signed1(lo)} to ${signed1(hi)} pts</td><td data-label="Reading">${esc(readPair(p))}</td></tr>`;
  }).join('');
  $('paired').querySelector('tbody').innerHTML = rows;
}

/* ---------- case explorer ---------- */
function applyFilters(announceResult) {
  const status = $('filter').value, wording = $('wording-filter').value;
  state.filtered = state.data.cases.filter(c => {
    const s = c.predictions.sft, d = c.predictions.dpo_clean;
    const heldOut = c.template === HELD_OUT_TEMPLATE;
    if (wording === 'seen' && heldOut) return false;
    if (wording === 'held' && !heldOut) return false;
    return status === 'all' || (status === 'success' && s.correct) || (status === 'action' && s.format_ok && !s.action_ok)
      || (status === 'evidence' && s.format_ok && s.action_ok && !s.evidence_ok) || (status === 'disagree' && s.correct !== d.correct);
  });
  state.page = 0; state.selectedId = state.filtered[0] ? state.filtered[0].row_id : null;
  renderList(); renderCase();
  if (announceResult) announce(`${state.filtered.length} cases match.`);
}

function renderList() {
  const pages = Math.max(1, Math.ceil(state.filtered.length / PER_PAGE));
  state.page = Math.max(0, Math.min(state.page, pages - 1));
  $('case-count').textContent = `${state.filtered.length} cases`;
  $('page-label').textContent = `${state.page + 1} / ${pages}`;
  setDisabled($('prev'), state.page === 0); setDisabled($('next'), state.page >= pages - 1);
  $('case-list').innerHTML = state.filtered.slice(state.page * PER_PAGE, (state.page + 1) * PER_PAGE).map(c =>
    `<li><button type="button" class="case-item" data-id="${esc(c.row_id)}"${c.row_id === state.selectedId ? ' aria-current="true"' : ''}><strong>${esc(c.row_id)}</strong><small>${esc(c.rule.replaceAll('_', ' '))} · ${wordingOf(c)}</small></button></li>`).join('')
    || '<li class="empty">No cases match.</li>';
}
const setDisabled = (btn, off) => { btn.setAttribute('aria-disabled', off ? 'true' : 'false'); };

function selectCase(id, { toPanel = false } = {}) {
  const idx = state.filtered.findIndex(c => c.row_id === id);
  if (idx < 0) return;
  state.selectedId = id;
  const wantPage = Math.floor(idx / PER_PAGE);
  if (wantPage !== state.page) { state.page = wantPage; renderList(); }
  else document.querySelectorAll('#case-list .case-item').forEach(b => { if (b.dataset.id === id) b.setAttribute('aria-current', 'true'); else b.removeAttribute('aria-current'); });
  renderCase();
  const c = state.filtered[idx];
  announce(`Case ${c.row_id}, ${wordingOf(c)}. ${NAMES[state.model]}: ${c.predictions[state.model].correct ? 'strict pass' : 'strict fail'}.`);
  if (toPanel || isNarrow()) { const h = $('case-title'); if (h) { h.focus({ preventScroll: true }); h.scrollIntoView({ block: 'start', behavior: scrollBehavior() }); } }
}

function renderCase() {
  const box = $('case-content');
  const c = state.filtered.find(x => x.row_id === state.selectedId);
  if (!c) { box.innerHTML = '<p class="loading">No cases match these filters.</p>'; return; }
  const lines = c.prompt.split('\n').map(line => {
    const id = (line.match(/^\[(L\d+)\]/) || [])[1];
    const ev = c.required_ids.includes(id);
    return `<div class="prompt-line${ev ? ' is-evidence' : ''}"><span class="ev-mark" aria-hidden="true">${ev ? '✓' : ''}</span><span>${ev ? '<span class="sr-only">Reference evidence: </span>' : ''}${esc(line)}</span></div>`;
  }).join('');
  box.innerHTML = `
    <div class="case-head"><div>
      <p class="eyebrow">Return request · ${esc(c.rule.replaceAll('_', ' '))}</p>
      <h3 id="case-title" tabindex="-1">Case ${esc(c.row_id)}</h3>
      <p class="case-meta">${esc(wordingOf(c))} · synthetic case</p></div>
      <button type="button" class="link-btn" data-back>Back to the case list</button></div>
    <div class="prompt-box"><h4>Policy and customer facts</h4><p class="legend"><span aria-hidden="true">✓</span> marks the lines that make up the reference evidence.</p>${lines}</div>
    <div class="reference"><span>Reference answer</span><strong>${esc(c.label)} · ${esc(c.required_ids.join(', '))}</strong></div>
    <p class="answer-title" id="resp-label">Saved model response</p>
    <div class="model-tabs" role="group" aria-labelledby="resp-label">${MODELS.map(m => `<button type="button" data-model="${m}" aria-pressed="${state.model === m}">${esc(NAMES[m])}</button>`).join('')}</div>
    <pre class="answer-box" id="answer-box"></pre>
    <ul class="verdict" id="verdict" aria-label="Checks for this response"></ul>
    <div class="twin" id="twin"></div>`;
  updateModel(false);
}

function updateModel(announceChange) {
  const c = state.filtered.find(x => x.row_id === state.selectedId); if (!c) return;
  const p = c.predictions[state.model];
  document.querySelectorAll('#case-content [data-model]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.model === state.model)));
  $('answer-box').textContent = p.output || '(empty response)';
  const pill = (ok, yes, no) => `<li class="${ok ? 'ok' : 'bad'}"><span aria-hidden="true">${ok ? '✓' : '×'}</span> ${ok ? yes : no}</li>`;
  $('verdict').innerHTML = pill(p.format_ok, 'Valid format', 'Invalid format') + pill(p.action_ok, 'Action correct', 'Action incorrect')
    + pill(p.evidence_ok, 'Evidence exact', 'Evidence incorrect') + pill(p.correct, 'Strict pass', 'Strict fail');
  const twin = state.data.cases.find(x => x.group_id === c.group_id && x.row_id !== c.row_id);
  $('twin').innerHTML = twin ? `<h4>Same case, other wording</h4><p>${esc(twin.row_id)} (${esc(wordingOf(twin))}): ${esc(NAMES[state.model])} scored <b>${twin.predictions[state.model].correct ? 'a strict pass' : 'a strict fail'}</b> here.</p><button type="button" class="link-btn" data-open="${esc(twin.row_id)}">Open that wording</button>` : '';
  if (announceChange) announce(`${NAMES[state.model]}: ${p.correct ? 'strict pass' : 'strict fail'}. ${p.action_ok ? 'Action correct' : 'Action incorrect'}, ${p.evidence_ok ? 'evidence exact' : 'evidence incorrect'}.`);
}

function openCase(id) {
  const idx = state.filtered.findIndex(c => c.row_id === id);
  if (idx < 0) {   // the other wording is hidden by the current filters: clear them so it can be shown
    $('filter').value = 'all'; $('wording-filter').value = 'all'; applyFilters(false);
  }
  selectCase(id, { toPanel: true });
}

function wireExplorer() {
  $('filter').addEventListener('change', () => applyFilters(true));
  $('wording-filter').addEventListener('change', () => applyFilters(true));
  const page = delta => btn => {
    if (btn.getAttribute('aria-disabled') === 'true') return;
    state.page += delta; state.selectedId = state.filtered[state.page * PER_PAGE].row_id; renderList(); renderCase();
    announce(`Page ${state.page + 1}. Showing case ${state.selectedId}.`);
  };
  $('prev').addEventListener('click', e => page(-1)(e.currentTarget)); $('next').addEventListener('click', e => page(1)(e.currentTarget));
  $('case-list').addEventListener('click', e => { const b = e.target.closest('.case-item'); if (b) selectCase(b.dataset.id); });
  $('case-content').addEventListener('click', e => {
    const m = e.target.closest('[data-model]'), o = e.target.closest('[data-open]'), back = e.target.closest('[data-back]');
    if (m) { state.model = m.dataset.model; updateModel(true); }
    else if (o) openCase(o.dataset.open);
    else if (back) { const cur = document.querySelector('#case-list [aria-current="true"]') || document.querySelector('#case-list .case-item'); if (cur) { cur.focus({ preventScroll: true }); cur.scrollIntoView({ block: 'center', behavior: scrollBehavior() }); } }
  });
}

function wireWording() {
  document.querySelectorAll('input[name="wording"]').forEach(r => r.addEventListener('change', () => {
    state.view = r.value; renderBars(); announce(`Showing ${VIEW_LABELS[state.view].toLowerCase()}.`);
  }));
}

/* ---------- presentation ---------- */
function buildSlides() {
  const d = state.data;
  const sft = d && metric('sft'), base = d && metric('base');
  const seen = d && metric('sft', 'seen_template').point, held = d && metric('sft', 'held_out_template').point;
  return [
    { kicker: 'The question', title: 'Can post-training make a small model decide well and show its evidence?', body: 'A fictional returns workflow tests one practical question: can a model choose ACCEPT, DENY or HOLD and point to the exact facts that support the choice?', note: 'A small learning experiment on synthetic data' },
    { kicker: 'The task', title: 'A decision is only as useful as its evidence.', body: 'Given a store policy and a customer ticket, the model returns an action and the line ids that justify it. A strict pass needs the right action and exactly the right evidence lines.', note: 'A rule oracle supplies the reference answer' },
    { kicker: 'What was tried', title: 'One task, four training paths.', body: 'Start from Qwen2.5-0.5B-Instruct. Compare the untrained model with supervised fine-tuning (SFT), DPO on preference pairs (clean and with 20% of labels flipped), and GRPO with a verifier reward. DPO and GRPO start from SFT.', note: 'One training seed per method' },
    { kicker: 'What the test found', title: 'SFT helped. The rest is not clearly different.', body: d ? `On the held-out test SFT scored ${pct(sft.point)} against ${pct(base.point)} for the untrained model. DPO and GRPO were not clearly different from SFT. New wording was harder: SFT scored ${pct(seen)} on seen wording and ${pct(held)} on held-out wording.` : 'Saved results are not loaded, so no numbers are shown here.', note: d ? `${sft.n_rows} test rows from ${groupsIn('all_rows')} groups, scored once after the models were frozen` : 'Serve this folder over HTTP to load the results' },
    { kicker: 'What is uncertain', title: 'An honest result is still a useful result.', body: 'One seed, a small model and synthetic labels limit what this shows. Tool-use training was not achieved. A fair next step is a stronger SFT start, several seeds per method, and a new validation set with a separately frozen test.', note: 'No claim that DPO or GRPO improved on SFT' }
  ];
}

let slides = buildSlides();
function renderSlide() {
  const s = slides[state.slide], n = slides.length;
  $('slide-counter').textContent = `Slide ${state.slide + 1} of ${n}`;
  $('slide-body').innerHTML = `<p class="slide-kicker">${esc(s.kicker)}</p><h3>${esc(s.title)}</h3><p>${esc(s.body)}</p><p class="slide-note">${esc(s.note)}</p>`;
  document.querySelectorAll('#slide-dots button').forEach((b, i) => { if (i === state.slide) b.setAttribute('aria-current', 'true'); else b.removeAttribute('aria-current'); });
  setDisabled($('slide-prev'), state.slide === 0); setDisabled($('slide-next'), state.slide === n - 1);
}
function goSlide(i) { const next = Math.max(0, Math.min(slides.length - 1, i)); if (next !== state.slide) { state.slide = next; renderSlide(); } }
function wireDeck() {
  $('slide-dots').innerHTML = slides.map((s, i) => `<button type="button" data-slide="${i}" aria-label="Slide ${i + 1}: ${esc(s.kicker)}"><i></i></button>`).join('');
  $('slide-dots').addEventListener('click', e => { const b = e.target.closest('[data-slide]'); if (b) goSlide(Number(b.dataset.slide)); });
  $('slide-prev').addEventListener('click', () => goSlide(state.slide - 1)); $('slide-next').addEventListener('click', () => goSlide(state.slide + 1));
  $('deck').addEventListener('keydown', e => {
    if (e.key === 'ArrowRight') { goSlide(state.slide + 1); e.preventDefault(); }
    else if (e.key === 'ArrowLeft') { goSlide(state.slide - 1); e.preventDefault(); }
    else if (e.key === 'Home') { goSlide(0); e.preventDefault(); }
    else if (e.key === 'End') { goSlide(slides.length - 1); e.preventDefault(); }
  });
  renderSlide();
}

/* ---------- views (Overview, System design, Presentation) ---------- */
let firstView = true;
function showView() {
  const hash = location.hash.slice(1) || 'overview';
  const view = VIEW_TITLES[hash] ? hash : (ANCHORS[hash] || 'overview');
  document.querySelectorAll('.view').forEach(v => { v.hidden = v.dataset.view !== view; });
  document.querySelectorAll('[data-view-link]').forEach(a => { if (a.dataset.viewLink === view) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current'); });
  document.title = `${VIEW_TITLES[view]} | Preference Lab`;
  if (ANCHORS[hash]) {
    const target = $(hash); requestAnimationFrame(() => { target.scrollIntoView({ behavior: scrollBehavior() }); const h = target.querySelector('h2'); if (h && !firstView) h.focus({ preventScroll: true }); });
  } else {
    window.scrollTo({ top: 0, behavior: 'auto' });
    if (!firstView) { const h = document.querySelector(`#${view} h1, #${view} h2`); if (h) h.focus({ preventScroll: true }); announce(`${VIEW_TITLES[view]} view.`); }
  }
  firstView = false;
}

/* ---------- start ---------- */
function failed() {
  const msg = 'Could not load the saved data. Serve this folder over HTTP and refresh.';
  for (const id of ['case-content', 'bars']) { const e = $(id); e.textContent = msg; e.setAttribute('role', 'alert'); }
  $('glance').innerHTML = `<li>${msg}</li>`; $('chart-sub').textContent = 'Saved results unavailable.'; $('case-count').textContent = 'No data';
}

document.querySelector('.skip-link').addEventListener('click', e => { e.preventDefault(); $('main').focus(); });
window.addEventListener('hashchange', showView);
wireDeck(); wireExplorer(); wireWording(); showView();

fetch('./data.json').then(r => { if (!r.ok) throw Error(`HTTP ${r.status}`); return r.json(); }).then(value => {
  state.data = value;
  renderGlance(); renderBars(); renderPaired(); applyFilters(false);
  slides = buildSlides(); renderSlide();
}).catch(failed);
