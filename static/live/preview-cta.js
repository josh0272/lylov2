(() => {
  const text = (selector, value) => {
    const node = document.querySelector(selector);
    if (node) node.textContent = value;
    return node;
  };

  const initSectionAnchors = () => {
    const intro = document.querySelector('.demo-intro');
    const warning = document.querySelector('.reg-trust');
    const demoHeading = document.querySelector('.demo-heading');

    if (intro?.id === 'demos') intro.removeAttribute('id');
    if (warning) warning.id = 'ai-duties';
    if (demoHeading) demoHeading.id = 'demos';
  };

  const initDesktopNav = () => {
    const headerInner = document.querySelector('.site-header .header-inner');
    const hamburger = document.getElementById('hamburger');
    if (!headerInner || !hamburger) return;

    let nav = headerInner.querySelector('.desktop-nav');
    if (!nav) {
      nav = document.createElement('nav');
      nav.className = 'desktop-nav';
      nav.setAttribute('aria-label', 'Primary');
      headerInner.insertBefore(nav, hamburger);
    }

    nav.innerHTML = `
      <a href="#ai-duties">AI & professional duties</a>
      <a href="#demos">Demos</a>
      <a href="#privacy">Private AI</a>
      <a class="desktop-pilot-link" href="/founding-pilot">Founding Pilot</a>
    `;
  };

  const closeMobileMenu = () => {
    const menu = document.getElementById('mobileMenu');
    const hamburger = document.getElementById('hamburger');
    if (!menu || !hamburger) return;
    menu.classList.remove('open');
    menu.setAttribute('aria-hidden', 'true');
    hamburger.setAttribute('aria-expanded', 'false');
    document.body.style.overflow = '';
  };

  const initMobileNav = () => {
    const panel = document.querySelector('#mobileMenu .panel-inner');
    if (!panel) return;

    panel.innerHTML = `
      <a href="#ai-duties">AI & professional duties</a>
      <a href="#demos">Demos</a>
      <a href="#privacy">Private AI</a>
      <a href="/founding-pilot">Founding Pilot</a>
    `;

    panel.querySelectorAll('a').forEach((link) => {
      link.addEventListener('click', closeMobileMenu);
    });
  };

  const initRegulatoryCopy = () => {
    const section = document.querySelector('.reg-trust');
    if (!section) return;

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
  };

  const initTeamResearch = () => {
    const finalSection = document.querySelector('.final');
    if (!finalSection || document.querySelector('.lylo-people')) return;

    const aboutPath = '/about';
    const joshPhoto = 'static/joshua-profile.jpg';
    const jessPhoto = 'static/jessica-profile.jpg';

    const people = document.createElement('section');
    people.className = 'lylo-people';
    people.innerHTML = `
      <div class="lylo-people-inner reveal in">
        <div class="lylo-section-kicker">Co-founders</div>
        <h3>Built by Joshua and Jessica.</h3>
        <p class="lylo-people-intro">A small team combining engineering and legal experience while Lylo is being researched, tested and shaped.</p>
        <div class="lylo-people-list" aria-label="People behind Lylo">
          <div class="lylo-person">
            <a class="lylo-person-photo" href="${aboutPath}#joshua" aria-label="Read more about Joshua Sam"><img src="${joshPhoto}" alt="Joshua Sam"></a>
            <div><strong>Joshua Sam</strong><span class="role">Product & Engineering</span><span class="credential">MEng Electrical & Mechanical Engineering · University of Strathclyde</span></div>
          </div>
          <div class="lylo-person">
            <a class="lylo-person-photo" href="${aboutPath}#jessica" aria-label="Read more about Jessica Jayan"><img src="${jessPhoto}" alt="Jessica Jayan"></a>
            <div><strong>Jessica Jayan</strong><span class="role">Legal Research & Workflow</span><span class="credential">Scots (Clinical) LLB · DPLP · University of Strathclyde</span></div>
          </div>
        </div>
        <a class="lylo-secondary-link" href="${aboutPath}">About the co-founders <span class="arrow">→</span></a>
      </div>`;

    const research = document.createElement('section');
    research.className = 'lylo-research-strip';
    research.innerHTML = `
      <div class="lylo-research-inner reveal in">
        <div class="lylo-section-kicker">Research</div>
        <h3>Help us build Lylo around real legal work.</h3>
        <p>We are speaking with people in law about the work that takes the most time, where AI could genuinely help, and what firms would need before trusting it. Our short questionnaire helps shape what Lylo should become.</p>
        <div class="lylo-research-action"><a class="lylo-secondary-link" href="/research">Take the questionnaire <span class="arrow">→</span></a></div>
        <div class="lylo-research-note">Around 5 minutes · used for product research.</div>
      </div>`;

    finalSection.parentNode.insertBefore(people, finalSection);
    finalSection.parentNode.insertBefore(research, finalSection);
  };

  const init = () => {
    initSectionAnchors();
    initDesktopNav();
    initMobileNav();
    initRegulatoryCopy();

    const heroCta = document.querySelector('.hero .cta');
    if (heroCta) {
      heroCta.textContent = 'Book a 20-minute demo';
      heroCta.setAttribute('href', '/founding-pilot#book');
    }
    text('.hero .note', 'See Lylo, ask questions, and decide if the pilot is worth testing.');
    text('.privacy-main', 'What if your firm could use the benefits of AI without sending sensitive client data to a public AI service? With Lylo, case files stay on the firm’s own systems.');

    const sections = Array.from(document.querySelectorAll('.demo-section'));
    const byHeading = (heading) => sections.find((section) => section.querySelector('.demo-copy h3')?.textContent.trim() === heading);

    const caseSection = byHeading('Ask the case. Get the answer.') || byHeading('Ask the case. Find the source.');
    const caseCta = caseSection?.querySelector('.demo-cta');
    if (caseCta) {
      caseCta.textContent = 'Explore the founding pilot';
      caseCta.setAttribute('href', '/founding-pilot');
    }

    const et1Section = byHeading('Turn case files into a completed form.') || byHeading('Turn case files into a draft form.');
    const et1Cta = et1Section?.querySelector('.demo-copy .demo-cta');
    if (et1Cta) {
      et1Cta.textContent = 'Explore the founding pilot';
      et1Cta.setAttribute('href', '/founding-pilot?workflow=et1');
    }

    const et1Form = et1Section?.querySelector('#et1-form-suggest');
    const et1Row = et1Form?.querySelector('.et1-suggest-row');
    if (et1Row && !et1Row.querySelector('.et1-book-demo')) {
      const demoLink = document.createElement('a');
      demoLink.className = 'et1-book-demo';
      demoLink.href = '/founding-pilot?workflow=et1#book';
      demoLink.textContent = 'Book a 20-minute demo';
      et1Row.appendChild(demoLink);
    } else {
      const demoLink = et1Row?.querySelector('.et1-book-demo');
      if (demoLink) demoLink.href = '/founding-pilot?workflow=et1#book';
    }

    const scheduleSection = byHeading('Know what the claim is worth.') || byHeading('Build a Schedule of Loss.');
    const scheduleCta = scheduleSection?.querySelector('.demo-cta');
    if (scheduleCta) {
      scheduleCta.textContent = 'Test this with your workflow';
      scheduleCta.setAttribute('href', '/founding-pilot?workflow=schedule#book');
    }

    const phoneSection = byHeading('Let Lylo answer the phone.');
    if (phoneSection) {
      if (!document.querySelector('.beyond-casework')) {
        const bridge = document.createElement('section');
        bridge.className = 'beyond-casework';
        bridge.innerHTML = `
          <div class="beyond-casework-inner">
            <div class="beyond-kicker">Beyond casework</div>
            <h3>AI can help with more than legal documents.</h3>
            <p>Alongside private legal workflows, Lylo can also be adapted for useful firm-wide tasks. One example is out-of-hours call handling.</p>
            <div class="beyond-boundary">The voice receptionist uses separate cloud voice services and is separate from Lylo’s private on-premise legal AI.</div>
          </div>
        `;
        phoneSection.parentNode.insertBefore(bridge, phoneSection);
      }

      const label = phoneSection.querySelector('.call-label');
      if (label) label.textContent = 'Try the live receptionist';

      phoneSection.querySelectorAll('.call-number').forEach((call) => {
        call.classList.add('phone-live-cta');
      });

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
      if (title) title.textContent = 'We are looking for a small number of law firms to pilot Lylo with us.';
      if (copy) copy.textContent = 'Start with a short demo. If it looks useful, test Lylo on one synthetic or properly anonymised matter and compare it with your normal workflow.';
      if (cta) {
        cta.textContent = 'Apply for the founding pilot';
        cta.setAttribute('href', '/founding-pilot#book');
      }
    }

    initTeamResearch();

    
  };

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init, { once:true });
  else init();
})();
