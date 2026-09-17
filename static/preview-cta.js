(() => {
  const applyScotlandPreview = () => {
    const desktopLink = document.querySelector('.desktop-nav a[href="#sra-warning"]');
    if (desktopLink) desktopLink.textContent = 'AI & professional duties';

    const mobileLink = document.querySelector('#mobileMenu .panel-inner a[href="#sra-warning"]');
    if (mobileLink) mobileLink.textContent = 'AI & professional duties';

    const section = document.querySelector('.reg-trust');
    if (!section) return;

    const heading = section.querySelector('.reg-question');
    const sourceLabel = section.querySelector('.reg-date');
    const lead = section.querySelector('.reg-lead');
    const cards = Array.from(section.querySelectorAll('.reg-card'));
    const line = section.querySelector('.reg-line');
    const sources = section.querySelector('.reg-sources');
    const carousel = section.querySelector('.reg-cards');

    if (heading) heading.textContent = 'AI can help. Your professional duties still matter.';
    if (sourceLabel) sourceLabel.textContent = 'Law Society of Scotland · Guide to Generative AI';
    if (lead) lead.textContent = 'For Scottish solicitors, the Law Society of Scotland highlights questions around accuracy and quality, client confidentiality, firm security, consent and oversight when generative AI is used. Lylo is being designed around those concerns.';
    if (carousel) carousel.setAttribute('aria-label', 'Key AI considerations for Scottish solicitors and how Lylo is being designed around them');

    const cardContent = [
      {
        title: 'Accuracy & quality',
        risk: 'Generative AI can produce inaccurate, incomplete or convincing-looking output.',
        lylo: 'Lylo is built to ground answers in the firm’s own documents and show the source material for review.'
      },
      {
        title: 'Confidentiality & security',
        risk: 'Client information and firm systems need careful protection when AI tools are used.',
        lylo: 'Lylo’s private legal AI is designed so sensitive case work can remain inside the firm’s environment.'
      },
      {
        title: 'Human oversight',
        risk: 'AI output still needs proper human review before it is relied on in legal work.',
        lylo: 'Lylo keeps the solicitor in control, with checking, editing and review built into the workflow.'
      },
      {
        title: 'Client consent & transparency',
        risk: 'Firms should consider when clients need to be informed or consent may be required for particular AI uses.',
        lylo: 'Lylo is designed as a controlled workflow so firms can decide where AI is used and how it fits their process.'
      },
      {
        title: 'Professional duties',
        risk: 'Using AI does not remove the professional duties that apply to solicitors and firms.',
        lylo: 'Lylo is an assistant to legal judgement, not a replacement for it.'
      }
    ];

    cards.forEach((card, index) => {
      const content = cardContent[index % cardContent.length];
      const title = card.querySelector('strong');
      const risk = card.querySelector('span');
      const lylo = card.querySelector('em');
      if (title) title.textContent = content.title;
      if (risk) risk.textContent = content.risk;
      if (lylo) lylo.textContent = content.lylo;
    });

    if (line) {
      line.innerHTML = 'Data protection obligations sit alongside professional duties. Firms still need to consider the tools, deployment and policies they use.<span class="reg-built">Lylo is being designed with privacy, control and review in mind from the start.</span>';
    }

    if (sources) {
      sources.innerHTML = '<a href="https://lawscot.org.uk/media/pl1lnu5n/ai-guide.pdf" target="_blank" rel="noopener">Law Society of Scotland: Guide to Generative AI</a> &nbsp;·&nbsp; <a href="https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/accountability-and-governance/guide-to-accountability-and-governance/data-protection-by-design-and-by-default/" target="_blank" rel="noopener">ICO: data protection by design</a> &nbsp;·&nbsp; <a href="https://media.sra.org.uk/solicitors/guidance/misuse-ai/" target="_blank" rel="noopener">England & Wales: SRA AI warning</a>';
    }
  };

  const runAfterLiveInit = () => {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', () => setTimeout(applyScotlandPreview, 0), { once: true });
    } else {
      setTimeout(applyScotlandPreview, 0);
    }
  };

  const script = document.createElement('script');
  script.src = '/live-assets/preview-cta.js?v=6';
  script.addEventListener('load', runAfterLiveInit, { once: true });
  document.head.appendChild(script);
})();