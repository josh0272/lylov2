(() => {
  const text = (selector, value) => {
    const node = document.querySelector(selector);
    if (node) node.textContent = value;
    return node;
  };

  const href = (selector, value) => {
    const node = document.querySelector(selector);
    if (node) node.setAttribute('href', value);
    return node;
  };

  const init = () => {
    const navLinks = document.querySelectorAll('#mobileMenu .panel-inner a');
    if (navLinks.length) {
      const last = navLinks[navLinks.length - 1];
      last.textContent = 'Founding Pilot';
      last.setAttribute('href', '/research');
    }

    const heroCta = document.querySelector('.hero .cta');
    if (heroCta) {
      heroCta.textContent = 'Book a 20-minute demo';
      heroCta.setAttribute('href', '/research');
    }
    text('.hero .note', 'See Lylo, ask questions, and decide if the pilot is worth testing.');

    const sections = Array.from(document.querySelectorAll('.demo-section'));
    const byHeading = (heading) => sections.find((section) => section.querySelector('.demo-copy h3')?.textContent.trim() === heading);

    const caseSection = byHeading('Ask the case. Get the answer.');
    const caseCta = caseSection?.querySelector('.demo-cta');
    if (caseCta) {
      caseCta.textContent = 'Explore the founding pilot';
      caseCta.setAttribute('href', '/research');
    }

    const scheduleSection = byHeading('Know what the claim is worth.');
    const scheduleCta = scheduleSection?.querySelector('.demo-cta');
    if (scheduleCta) {
      scheduleCta.textContent = 'Test this with your workflow';
      scheduleCta.setAttribute('href', '/research');
    }

    const phoneSection = byHeading('Let Lylo answer the phone.');
    if (phoneSection) {
      const label = phoneSection.querySelector('.call-label');
      if (label) label.textContent = 'Try the live receptionist';

      const extraCta = phoneSection.querySelector('.demo-copy > .demo-cta');
      if (extraCta) extraCta.remove();

      const helper = document.createElement('div');
      helper.className = 'phone-call-helper';
      helper.textContent = 'Call Lylo and speak as if you were a client.';
      const callBox = phoneSection.querySelector('.call-box');
      if (callBox && !phoneSection.querySelector('.phone-call-helper')) callBox.appendChild(helper);
    }

    const final = document.querySelector('.final .reveal');
    if (final) {
      const title = final.querySelector('h3');
      const copy = final.querySelector('p');
      const cta = final.querySelector('.cta');
      if (title) title.textContent = 'Put Lylo to work on a real workflow.';
      if (copy) copy.textContent = 'Start with a short demo. If it looks useful, test Lylo against a synthetic or properly anonymised matter from your firm.';
      if (cta) {
        cta.textContent = 'Apply for the founding pilot';
        cta.setAttribute('href', '/research');
      }
    }

    if (!document.getElementById('preview-cta-styles')) {
      const style = document.createElement('style');
      style.id = 'preview-cta-styles';
      style.textContent = `
        .phone-call-helper{margin-top:11px;color:#77879a;font-size:11px;line-height:1.45}
        @media(max-width:979px){.phone-call-helper{margin-top:10px;font-size:10.5px}}
      `;
      document.head.appendChild(style);
    }
  };

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init, { once:true });
  else init();
})();