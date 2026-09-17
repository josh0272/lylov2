(() => {
  const run = () => {
    const hero = document.querySelector('.hero');
    const heroCta = hero?.querySelector('.cta');
    if (hero && heroCta) {
      const note = hero.querySelector('.note');
      if (note) {
        note.textContent = 'No preparation. No client data. No commitment. See the demos, ask questions and decide whether Lylo is worth exploring further.';
      }
    }

    const final = document.querySelector('.final .reveal');
    if (final && !final.querySelector('.lylo-pilot-brief')) {
      const brief = document.createElement('div');
      brief.className = 'lylo-pilot-brief';
      brief.setAttribute('aria-label', 'Founding pilot steps');
      brief.innerHTML = `
        <div class="lylo-pilot-step">
          <span class="lylo-pilot-step-no">01</span>
          <strong>See Lylo live</strong>
          <span>Start with a short 20–30 minute demonstration of the current system and demos.</span>
        </div>
        <div class="lylo-pilot-step">
          <span class="lylo-pilot-step-no">02</span>
          <strong>Test a safe workflow</strong>
          <span>Use synthetic or properly anonymised material and compare Lylo with your normal process.</span>
        </div>
        <div class="lylo-pilot-step">
          <span class="lylo-pilot-step-no">03</span>
          <strong>Review the result</strong>
          <span>Review what worked and only discuss a paid pilot if the experiment proves useful.</span>
        </div>`;

      const cta = final.querySelector('.cta');
      if (cta) cta.insertAdjacentElement('beforebegin', brief);
      else final.appendChild(brief);
    }

    if (final && !final.querySelector('.lylo-demo-explainer')) {
      const cta = final.querySelector('.cta');
      const explainer = document.createElement('p');
      explainer.className = 'lylo-demo-explainer';
      explainer.textContent = 'In the demo: see the document Q&A, ET1 and Schedule of Loss workflows, hear how the private/on-premise system is being designed, and tell us where it would—or wouldn’t—fit your firm.';
      if (cta) cta.insertAdjacentElement('afterend', explainer);
      else final.appendChild(explainer);
    }

    if (!document.getElementById('lylo-pilot-polish-styles')) {
      const style = document.createElement('style');
      style.id = 'lylo-pilot-polish-styles';
      style.textContent = `
        .lylo-demo-explainer{max-width:720px;margin:11px auto 0;color:#748398;font-size:11.5px;line-height:1.58;text-align:center}
        .lylo-pilot-brief{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));width:min(100%,820px);margin:30px auto 28px;border-top:1px solid rgba(255,255,255,.065);border-bottom:1px solid rgba(255,255,255,.065)}
        .lylo-pilot-step{position:relative;padding:19px 24px;text-align:left}
        .lylo-pilot-step+.lylo-pilot-step{border-left:1px solid rgba(255,255,255,.055)}
        .lylo-pilot-step-no{display:block;margin-bottom:8px;color:#7896bb;font-size:9px;font-weight:700;letter-spacing:.13em}
        .lylo-pilot-step strong{display:block;margin-bottom:6px;color:#e7edf5;font-size:12.5px;font-weight:600;line-height:1.35}
        .lylo-pilot-step>span:last-child{display:block;color:#7e8da1;font-size:10.5px;line-height:1.5}
        @media(max-width:700px){
          .lylo-demo-explainer{max-width:340px;margin-top:10px;font-size:10.5px;line-height:1.55}
          .lylo-pilot-brief{grid-template-columns:1fr;width:min(100%,350px);margin:26px auto 24px}
          .lylo-pilot-step{padding:15px 4px;text-align:center}
          .lylo-pilot-step+.lylo-pilot-step{border-left:0;border-top:1px solid rgba(255,255,255,.055)}
          .lylo-pilot-step-no{margin-bottom:6px}
          .lylo-pilot-step strong{font-size:12px}
          .lylo-pilot-step>span:last-child{max-width:300px;margin:0 auto;font-size:10.5px}
        }
      `;
      document.head.appendChild(style);
    }
  };

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', run, { once: true });
  else run();
})();