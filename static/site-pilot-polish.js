(() => {
  const run = () => {
    const hero = document.querySelector('.hero');
    const heroCta = hero?.querySelector('.cta');
    if (hero && heroCta) {
      const note = hero.querySelector('.note');
      if (note) {
        note.textContent = 'No preparation. No client data. No commitment. See the demos, ask questions and decide whether Lylo is worth exploring further.';
      }

      const subheading = hero.querySelector('.bullet');
      if (subheading) {
        subheading.textContent = 'See how Lylo is being designed to draft Statements of Fact, ET1s and Schedules of Loss — without sending client data to a public AI service.';
      }
    }

    const desktopNav = document.querySelector('.desktop-nav');
    if (desktopNav && !desktopNav.querySelector('a[href="/research"]')) {
      const researchLink = document.createElement('a');
      researchLink.href = '/research';
      researchLink.textContent = 'Research';
      const pilotLink = desktopNav.querySelector('.desktop-pilot-link');
      if (pilotLink) desktopNav.insertBefore(researchLink, pilotLink);
      else desktopNav.appendChild(researchLink);
    }

    const mobilePanel = document.querySelector('#mobileMenu .panel-inner');
    if (mobilePanel && !mobilePanel.querySelector('a[href="/research"]')) {
      const researchLink = document.createElement('a');
      researchLink.href = '/research';
      researchLink.textContent = 'Research';
      const pilotLink = Array.from(mobilePanel.querySelectorAll('a')).find((link) => link.getAttribute('href') === '/founding-pilot');
      if (pilotLink) mobilePanel.insertBefore(researchLink, pilotLink);
      else mobilePanel.appendChild(researchLink);
    }

    document.querySelector('.lylo-research-strip')?.remove();

    const bookingHref = '/founding-pilot#book';
    const demoHeadings = new Set([
      'Ask the case. Get the answer.',
      'Ask the case. Find the source.',
      'Turn case files into a completed form.',
      'Turn case files into a draft form.',
      'Know what the claim is worth.',
      'Build a Schedule of Loss.'
    ]);

    document.querySelectorAll('.demo-section').forEach((section) => {
      const heading = section.querySelector('.demo-copy h3')?.textContent.trim();
      if (!demoHeadings.has(heading)) return;

      const cta = section.querySelector('.demo-copy .demo-cta');
      if (!cta) return;

      if (cta.tagName === 'BUTTON') {
        const link = document.createElement('a');
        link.className = Array.from(cta.classList).filter((name) => name !== 'et1-suggest-jump').join(' ');
        link.href = bookingHref;
        link.textContent = 'Book a 20-minute demo';
        cta.replaceWith(link);
      } else {
        cta.textContent = 'Book a 20-minute demo';
        cta.setAttribute('href', bookingHref);
      }
    });

    const final = document.querySelector('.final .reveal');
    if (final) {
      const finalCopy = final.querySelector('p');
      if (finalCopy) {
        finalCopy.textContent = 'The demos shown above are proof-of-concept demonstrations. In a short demo, see the document Q&A, ET1 and Schedule of Loss workflows, hear how the private/on-premise system is being designed, and tell us where it would—or wouldn’t—fit your firm. If it looks useful, work with us towards testing Lylo on one synthetic or properly anonymised matter and comparing it with your normal workflow.';
      }

      final.querySelector('.lylo-demo-explainer')?.remove();

      const finalCta = final.querySelector('.cta');
      if (finalCta) {
        finalCta.textContent = 'Explore the founding pilot';
        finalCta.setAttribute('href', '/founding-pilot');
      }
    }

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

    if (!document.getElementById('lylo-pilot-polish-styles')) {
      const style = document.createElement('style');
      style.id = 'lylo-pilot-polish-styles';
      style.textContent = `
        .lylo-pilot-brief{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));width:min(100%,820px);margin:30px auto 28px;border-top:1px solid rgba(255,255,255,.065);border-bottom:1px solid rgba(255,255,255,.065)}
        .lylo-pilot-step{position:relative;padding:19px 24px;text-align:left}
        .lylo-pilot-step+.lylo-pilot-step{border-left:1px solid rgba(255,255,255,.055)}
        .lylo-pilot-step-no{display:block;margin-bottom:8px;color:#7896bb;font-size:9px;font-weight:700;letter-spacing:.13em}
        .lylo-pilot-step strong{display:block;margin-bottom:6px;color:#e7edf5;font-size:12.5px;font-weight:600;line-height:1.35}
        .lylo-pilot-step>span:last-child{display:block;color:#7e8da1;font-size:10.5px;line-height:1.5}
        @media(max-width:700px){
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