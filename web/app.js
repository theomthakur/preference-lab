const names = {base:'Base model',sft:'SFT',dpo_clean:'DPO · clean',dpo_corrupt:'DPO · flipped',grpo:'GRPO'};
const descriptions = {base:'Before training',sft:'Supervised examples',dpo_clean:'Clean preferences',dpo_corrupt:'20% labels flipped',grpo:'Verifier reward'};
let data, filtered=[], page=0, selectedId=null, model='sft';
const perPage=7;
const el=id=>document.getElementById(id);
const escapeHtml=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const percent=n=>(n*100).toFixed(2).replace(/\.?0+$/, '')+'%';
function renderBars(){
  el('bars').innerHTML=Object.keys(names).map(key=>{const v=data.metrics.models[key].all_rows.point;return `<div class="bar-row"><div class="bar-label">${names[key]}<small>${descriptions[key]}</small></div><div class="bar-track"><div class="bar-fill ${key}" style="width:${v*100}%"></div></div><div class="bar-value">${percent(v)}</div></div>`}).join('');
}
function filterCases(){
 const value=el('filter').value;
 filtered=data.cases.filter(c=>{const s=c.predictions.sft,d=c.predictions.dpo_clean;return value==='all'||(value==='success'&&s.correct)||(value==='action'&&s.format_ok&&!s.action_ok)||(value==='evidence'&&s.format_ok&&s.action_ok&&!s.evidence_ok)||(value==='disagree'&&s.correct!==d.correct)});
 page=0; selectedId=filtered[0]?.row_id||null; renderList();renderCase();
}
function renderList(){
 const pages=Math.max(1,Math.ceil(filtered.length/perPage));page=Math.max(0,Math.min(page,pages-1));
 el('case-count').textContent=filtered.length+' CASES';el('page-label').textContent=`${page+1} / ${pages}`;el('prev').disabled=page===0;el('next').disabled=page>=pages-1;
 el('case-list').innerHTML=filtered.slice(page*perPage,(page+1)*perPage).map(c=>`<button class="case-item ${c.row_id===selectedId?'active':''}" data-id="${escapeHtml(c.row_id)}"><strong>${escapeHtml(c.row_id)}</strong><small>${escapeHtml(c.rule.replaceAll('_',' '))} · ${c.template===3?'held-out wording':'seen wording'}</small></button>`).join('');
 el('case-list').querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>{selectedId=b.dataset.id;renderList();renderCase()}));
}
function renderCase(){
 const c=filtered.find(x=>x.row_id===selectedId);if(!c){el('case-content').innerHTML='<p class="loading">No cases match this filter.</p>';return}
 const p=c.predictions[model];
 const lines=c.prompt.split('\n').map(line=>{const id=line.match(/^\[(L\d+)\]/)?.[1];return `<div class="prompt-line ${c.required_ids.includes(id)?'evidence':''}">${escapeHtml(line)}</div>`}).join('');
 const status=(ok,yes,no)=>`<span class="${ok?'':'bad'}">${ok?'✓':'×'} ${ok?yes:no}</span>`;
 el('case-content').innerHTML=`<div class="case-head"><div><div class="eyebrow">RETURN REQUEST / ${escapeHtml(c.rule.replaceAll('_',' '))}</div><h3>A decision, with evidence.</h3><p>${escapeHtml(c.row_id)} · ${c.template===3?'HELD-OUT WORDING':'SEEN WORDING'}</p></div><span class="case-note">SYNTHETIC CASE</span></div><div class="prompt-box"><h4>POLICY + CUSTOMER FACTS <span>· HIGHLIGHTED LINES ARE THE REFERENCE EVIDENCE</span></h4>${lines}</div><div class="reference"><span>REFERENCE ANSWER</span><strong>${escapeHtml(c.label)} · ${escapeHtml(c.required_ids.join(', '))}</strong></div><div class="answer-title">SAVED MODEL RESPONSE</div><div class="model-tabs">${Object.keys(names).map(k=>`<button data-model="${k}" class="${model===k?'active':''}" aria-pressed="${model===k}">${names[k]}</button>`).join('')}</div><div class="answer-box">${escapeHtml(p.output||'(empty response)')}</div><div class="verdict">${status(p.format_ok,'Valid format','Invalid format')}${status(p.action_ok,'Action correct','Action incorrect')}${status(p.evidence_ok,'Evidence exact','Evidence incorrect')}${status(p.correct,'Strict pass','Strict fail')}</div>`;
 el('case-content').querySelectorAll('[data-model]').forEach(b=>b.addEventListener('click',()=>{model=b.dataset.model;renderCase()}));
}
el('filter').addEventListener('change',filterCases);el('prev').addEventListener('click',()=>{page--;selectedId=filtered[page*perPage].row_id;renderList();renderCase()});el('next').addEventListener('click',()=>{page++;selectedId=filtered[page*perPage].row_id;renderList();renderCase()});
fetch('./data.json').then(r=>{if(!r.ok)throw Error(`HTTP ${r.status}`);return r.json()}).then(value=>{data=value;renderBars();filterCases()}).catch(()=>{el('case-content').innerHTML='<p class="loading">Could not load saved data. Serve this folder over HTTP and refresh.</p>';el('bars').textContent='Saved results unavailable.'});
