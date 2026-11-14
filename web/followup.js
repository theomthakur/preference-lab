/* The follow-up is a separate frozen experiment. Values come from its saved metrics. */
fetch('./followup.json')
  .then(response => { if (!response.ok) throw new Error('Results unavailable'); return response.json(); })
  .then(data => {
    window.preferenceFollowup = data;
    window.dispatchEvent(new Event('preference-followup-ready'));
    const {metrics} = data;
    const m = metrics.overall, b = metrics.group_bootstrap;
    const pct = n => (n * 100).toFixed(2).replace(/\.?0+$/, '') + '%';
    const items = [
      ['Correct decision', pct(m.action_rate), 'Action only'],
      ['Exact evidence', pct(m.evidence_rate), 'Evidence only'],
      ['Strict accuracy', pct(m.correct_rate), `${m.correct} / ${m.n} · action and evidence`],
      ['Held-out wording', pct(b.held_out_wording.strict_accuracy), 'New wording · strict accuracy']
    ];
    const target = document.getElementById('followup-results');
    target.replaceChildren(...items.map(([label, value, detail]) => {
      const card = document.createElement('article'); card.className = 'followup-card';
      const name = document.createElement('span'); name.textContent = label;
      const number = document.createElement('strong'); number.textContent = value;
      const foot = document.createElement('small'); foot.textContent = detail;
      card.append(name, number, foot); return card;
    }));
  })
  .catch(() => { document.getElementById('followup-results').textContent = 'Follow-up results could not be loaded. Serve this site over HTTP.'; });
