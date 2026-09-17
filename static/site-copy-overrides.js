(() => {
  const applyCopy = () => {
    document.querySelectorAll('.desktop-nav a[href="#sra-warning"], #mobileMenu .panel-inner a[href="#sra-warning"]').forEach(link => {
      link.textContent = 'AI & professional duties';
    });

    const section = document.querySelector('.reg-trust');
    if (section) {
      const heading = section.querySelector('.reg-question');
      const sourceLabel = section.querySelector('.reg-date');
      const lead = section.querySelector('.reg-lead');
      const cards = Array.from(section.querySelectorAll('.reg-card'));
      const line = section.querySelector('.reg-line');
      const sources = section.querySelector('.reg-sources');
      const carousel = section.querySelector('.reg-cards');

      if (heading) heading.textContent = 'AI use in law is changing. Is your firm ready?';
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
          lylo: 'Lylo supports <u>your</u> informed decisions.'
        }
      ];

      cards.forEach((card, index) => {
        const content = cardContent[index % cardContent.length];
        const title = card.querySelector('strong');
        const risk = card.querySelector('span');
        const lylo = card.querySelector('em');
        if (title) title.textContent = content.title;
        if (risk) risk.textContent = content.risk;
        if (lylo) {
          if (index % cardContent.length === 4) lylo.innerHTML = content.lylo;
          else lylo.textContent = content.lylo;
        }
      });

      if (line) {
        line.innerHTML = 'UK GDPR works alongside these professional responsibilities: data protection by design, data minimisation and security still apply.<span class="reg-built">Lylo is designed with these responsibilities in mind from the beginning.</span>';
      }

      if (sources) {
        sources.innerHTML = '<a href="https://lawscot.org.uk/media/pl1lnu5n/ai-guide.pdf" target="_blank" rel="noopener">Law Society of Scotland: Guide to Generative AI</a> &nbsp;·&nbsp; <a href="https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/accountability-and-governance/guide-to-accountability-and-governance/data-protection-by-design-and-by-default/" target="_blank" rel="noopener">ICO: data protection by design</a> &nbsp;·&nbsp; <a href="https://media.sra.org.uk/solicitors/guidance/misuse-ai/" target="_blank" rel="noopener">England & Wales: SRA AI warning</a>';
      }
    }

    const privacyMain = document.querySelector('.privacy-main');
    if (privacyMain) {
      privacyMain.textContent = 'What if your firm could use the benefits of AI without sending sensitive client data to a public AI service? With Lylo, case files stay on the firm’s own systems.';
    }

    const peopleSection = document.querySelector('.lylo-people');
    if (peopleSection) {
      const kicker = peopleSection.querySelector('.lylo-section-kicker');
      if (kicker) kicker.textContent = 'Co-founders';

      peopleSection.querySelectorAll('.lylo-person').forEach(person => {
        const name = person.querySelector('strong')?.textContent.trim();
        const role = person.querySelector('.role');
        if (!role) return;
        if (name === 'Joshua Sam') role.textContent = 'Product & Engineering';
        if (name === 'Jessica Jayan') role.textContent = 'Legal Research & Workflow';
      });

      const aboutLink = peopleSection.querySelector('.lylo-secondary-link');
      if (aboutLink) aboutLink.innerHTML = 'About the co-founders <span class="arrow">→</span>';
    }
  };

  let attempts = 0;
  const timer = setInterval(() => {
    applyCopy();
    attempts += 1;
    if (attempts >= 25) clearInterval(timer);
  }, 120);

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', applyCopy, { once: true });
  else applyCopy();

  const observer = new MutationObserver(() => applyCopy());
  observer.observe(document.documentElement, { childList: true, subtree: true });
  setTimeout(() => observer.disconnect(), 5000);
})();