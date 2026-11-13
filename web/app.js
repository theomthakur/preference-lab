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

const slides = [
  {kicker:'01 / THE QUESTION',title:'Can preference training improve a small model?',body:'We use a fictional returns workflow to study a practical question: can a model make the right decision and point to the facts that support it?',note:'A small learning experiment · synthetic data'},
  {kicker:'02 / THE TASK',title:'A decision is only as useful as its evidence.',body:'Given a store policy and a customer request, the model must return ACCEPT, DENY or HOLD and cite the exact lines that justify its choice.',note:'Strict pass = correct action + exact evidence'},
  {kicker:'03 / THE APPROACH',title:'One task. Several training paths.',body:'Start with Qwen2.5-0.5B-Instruct. Compare the base model with supervised fine-tuning, DPO on preference pairs, and GRPO with verifier rewards.',note:'DPO and GRPO start from the SFT model'},
  {kicker:'04 / THE RESULT',title:'SFT helped. The rest is uncertain.',body:'On the held-out test, SFT was correct on 11.75% of rows versus 0% for the base. DPO and GRPO did not establish a clear gain over SFT.',note:'800 rows · 400 groups · one training seed'},
  {kicker:'05 / THE TAKEAWAY',title:'An honest result is still a useful result.',body:'Inspect failures, keep the limitations visible, and design the next experiment before making stronger claims. Tool-use training was not achieved.',note:'Synthetic labels · small model · no production claims'}
];
let slideIndex=0;
function renderSlide(){const slide=slides[slideIndex];el('slide-counter').textContent=String(slideIndex+1).padStart(2,'0')+' / 05';el('slide-body').innerHTML=`<div class="slide-kicker">${slide.kicker}</div><h3>${slide.title}</h3><p>${slide.body}</p><div class="slide-note">✳ &nbsp; ${slide.note}</div>`;el('slide-dots').innerHTML=slides.map((_,i)=>`<button class="${i===slideIndex?'active':''}" data-slide="${i}" aria-label="Go to slide ${i+1}" aria-current="${i===slideIndex?'true':'false'}"></button>`).join('');el('slide-prev').disabled=slideIndex===0;el('slide-next').disabled=slideIndex===slides.length-1;el('slide-dots').querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>{slideIndex=Number(b.dataset.slide);renderSlide()}))}
el('slide-prev').addEventListener('click',()=>{slideIndex=Math.max(0,slideIndex-1);renderSlide()});el('slide-next').addEventListener('click',()=>{slideIndex=Math.min(slides.length-1,slideIndex+1);renderSlide()});renderSlide();
