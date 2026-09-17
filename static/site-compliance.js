(() => {
  const applyDynamicCopy = () => {
    const people = document.querySelector('.lylo-people');
    if (people) {
      const kicker = people.querySelector('.lylo-section-kicker');
      if (kicker) kicker.textContent = 'Meet the people behind Lylo';

      const link = people.querySelector('.lylo-secondary-link');
      if (link) link.innerHTML = 'Meet the co-founders of Lylo <span class="arrow">→</span>';
    }

    const research = document.querySelector('.lylo-research-strip');
    if (research) {
      const heading = research.querySelector('h3');
      const body = research.querySelector('p');
      const note = research.querySelector('.lylo-research-note');
      if (heading) heading.textContent = 'Help us develop Lylo around real legal work.';
      if (body) body.textContent = 'We are speaking with people in law about the work that takes the most time, where AI could genuinely help, and what firms would need before trusting it. Our short questionnaire helps guide Lylo’s development.';
      if (note) note.textContent = 'Around 5 minutes · Used for product research.';
    }
  };

  const run = () => {
    const reg = document.querySelector('.reg-trust');
    if (reg) {
      const tag = reg.querySelector('.reg-date');
      if (tag) {
        tag.innerHTML = '<span class="reg-pill">Law Society of Scotland · Guide to Generative AI</span><span class="reg-pill reg-pill-secondary">England & Wales · 17 Aug 2026 · SRA Misuse of AI warning</span>';
        tag.classList.add('reg-date-pills');
      }

      const line = reg.querySelector('.reg-line');
      if (line) {
        line.innerHTML = 'UK GDPR works alongside these professional responsibilities. The wider UK framework also includes the Data Protection Act 2018, the Data (Use and Access) Act 2025, and PECR 2003 where relevant.<span class="reg-built">Lylo is designed with these responsibilities in mind from the beginning.</span>';
      }
    }

    applyDynamicCopy();
    setTimeout(applyDynamicCopy, 50);
    setTimeout(applyDynamicCopy, 200);
    setTimeout(applyDynamicCopy, 600);

    const footer = document.querySelector('body > footer');
    if (footer && !footer.querySelector('.lylo-privacy-link')) {
      const privacy = document.createElement('a');
      privacy.className = 'lylo-privacy-link';
      privacy.href = '/static/privacy.html';
      privacy.textContent = 'Privacy';
      privacy.style.cssText = 'display:inline-block;margin-right:14px;color:#8da0b8;text-decoration:none;';
      footer.insertBefore(privacy, footer.firstChild);
    }

    if (!document.getElementById('lylo-compliance-styles')) {
      const style = document.createElement('style');
      style.id = 'lylo-compliance-styles';
      style.textContent = `
        .reg-date-pills{display:flex!important;justify-content:center;align-items:center;gap:8px;flex-wrap:wrap;text-transform:none!important;letter-spacing:0!important}
        .reg-pill{display:inline-flex;align-items:center;min-height:28px;padding:0 10px;border:1px solid rgba(99,168,255,.18);border-radius:999px;background:rgba(36,69,108,.16);color:#9fc8fa;font-size:10px;font-weight:600;letter-spacing:.02em}
        .reg-pill-secondary{border-color:rgba(255,255,255,.10);background:rgba(255,255,255,.025);color:#9aa9bc}
        @media(max-width:700px){.reg-date-pills{gap:6px}.reg-pill{min-height:26px;padding:0 9px;font-size:9px;text-align:center}}
      `;
      document.head.appendChild(style);
    }
  };

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', run, { once: true });
  else run();
})();