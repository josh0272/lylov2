(function livePolish(){
  const run = () => {
    const bookingHref = '/call#book';
    const bookingLabel = 'See Lylo in a 20-minute call';

    const hero = document.querySelector('.hero');
    if (hero) {
      const tagline = hero.querySelector('.tagline');
      const value = hero.querySelector('.bullet');
      const cta = hero.querySelector('.cta');
      const note = hero.querySelector('.note');
      if (tagline) tagline.textContent = 'Private AI, designed for solicitors.';
      if (value) value.textContent = 'Search case files, prepare legal drafts and reduce repetitive work—while keeping confidential client information under your firm’s control.';
      if (cta) {
        cta.textContent = bookingLabel;
        cta.href = bookingHref;
      }
      if (note) {
        note.innerHTML = '<strong>No preparation. No client data. No sales commitment.</strong><span>See the most relevant working demonstration and identify one process in your firm that may suit a safe AI test.</span>';
      }
    }

    const orderNav = (nav, mobile = false) => {
      if (!nav) return;
      const find = (href) => Array.from(nav.querySelectorAll('a')).find((a) => a.getAttribute('href') === href);
      let research = find('/research');
      if (!research) {
        research = document.createElement('a');
        research.href = '/research';
        research.textContent = 'Research';
      }
      let next = nav.querySelector('.lylo-pilot-nav-text') || find('/call');
      if (!next) next = document.createElement('a');
      next.className = 'lylo-pilot-nav-text';
      next.href = '/call';
      next.textContent = 'What happens on the call';
      let book = nav.querySelector('.desktop-pilot-link, .lylo-mobile-booking-link');
      if (!book) book = document.createElement('a');
      book.className = mobile ? 'lylo-mobile-booking-link' : 'desktop-pilot-link';
      book.href = bookingHref;
      book.textContent = '20-minute call';
      [find('#demos'), find('#privacy'), find('#ai-duties'), research, next, book].forEach((link) => {
        if (link) nav.appendChild(link);
      });
    };
    orderNav(document.querySelector('.desktop-nav'));
    orderNav(document.querySelector('#mobileMenu .panel-inner'), true);
    document.querySelector('.lylo-research-strip')?.remove();

    const demoIntro = document.querySelector('.demo-intro');
    const demoHeading = document.querySelector('.demo-heading');
    if (demoIntro) demoIntro.classList.add('lylo-demos-first');
    if (demoHeading) {
      const title = demoHeading.querySelector('h2');
      const copy = demoHeading.querySelector('p');
      if (title) title.textContent = 'See Lylo work on real legal tasks.';
      if (copy) copy.textContent = 'Three working proof-of-concept demonstrations show how Lylo could search, draft and calculate inside a private, firm-controlled system.';
    }

    const sections = Array.from(document.querySelectorAll('.demo-section'));
    const findSection = (terms) => sections.find((section) => {
      const heading = section.querySelector('.demo-copy h3')?.textContent.toLowerCase() || '';
      return terms.some((term) => heading.includes(term));
    });
    const caseSection = findSection(['ask the case']);
    const et1Section = findSection(['draft form', 'completed form', 'et1']);
    const scheduleSection = findSection(['schedule of loss', 'claim is worth']);
    const phoneSection = sections.find((section) => section.classList.contains('phone-section'));

    const updateDemo = (section, heading, copy) => {
      if (!section) return;
      const title = section.querySelector('.demo-copy h3');
      const body = section.querySelector('.demo-copy p');
      const cta = section.querySelector('.demo-copy .demo-cta');
      if (title) title.textContent = heading;
      if (body) body.textContent = copy;
      if (cta) {
        if (cta.tagName === 'BUTTON') {
          const link = document.createElement('a');
          link.className = 'demo-cta';
          link.href = bookingHref;
          link.textContent = 'See this in a 20-minute call';
          cta.replaceWith(link);
        } else {
          cta.href = bookingHref;
          cta.textContent = 'See this in a 20-minute call';
        }
      }
    };
    updateDemo(caseSection, 'Ask the case. Find the source.', 'Search across a case file, bring the relevant answer into one place and open the source behind it before relying on the result.');
    updateDemo(et1Section, 'Turn case documents into an ET1 draft.', 'Pull names, dates and case details from uploaded documents, prepare the ET1 and flag anything that still needs the solicitor’s review.');
    updateDemo(scheduleSection, 'Build and check a Schedule of Loss.', 'Bring pay, dates and losses into a structured draft, show the calculation behind each figure and keep the supporting source available for review.');

    let other = document.querySelector('.lylo-other-workflows');
    const extension = document.querySelector('.et1-extension');
    if (extension && !other) {
      other = document.createElement('section');
      other.className = 'lylo-other-workflows';
      const inner = document.createElement('div');
      inner.className = 'lylo-other-workflows-inner reveal in';
      other.appendChild(inner);
      inner.appendChild(extension);
    }
    if (other) {
      const title = other.querySelector('.et1-extension-head h4');
      const copy = other.querySelector('.et1-extension-head p');
      if (title) title.textContent = 'Other forms and workflows.';
      if (copy) copy.textContent = 'These demonstrations are a starting point. Lylo could be developed around any repeatable process or form used by your firm.';
      let modulesLabel = other.querySelector('.lylo-modules-label');
      if (!modulesLabel) {
        modulesLabel = document.createElement('p');
        modulesLabel.className = 'lylo-modules-label';
        if (copy) copy.insertAdjacentElement('afterend', modulesLabel);
        else other.querySelector('.et1-extension-head')?.appendChild(modulesLabel);
      }
      modulesLabel.textContent = 'Potential form or workflow modules include:';
    }

    let mission = document.querySelector('.lylo-mission');
    if (!mission) {
      mission = document.createElement('section');
      mission.className = 'lylo-mission';
    }
    mission.innerHTML = '<div class="lylo-mission-inner"><div class="lylo-mission-kicker">Our mission</div><h2>Make secure, firm-controlled AI practical for law firms of every size.</h2></div>';

    const privacy = document.querySelector('.privacy');
    const lastPrincipal = scheduleSection || et1Section || caseSection;
    if (lastPrincipal && other) lastPrincipal.insertAdjacentElement('afterend', other);
    if (other) other.insertAdjacentElement('afterend', mission);
    else if (lastPrincipal) lastPrincipal.insertAdjacentElement('afterend', mission);
    if (mission && privacy) mission.insertAdjacentElement('afterend', privacy);

    if (privacy) {
      const heading = privacy.querySelector('h3');
      const intro = privacy.querySelector('.privacy-main');
      if (heading) heading.textContent = 'Private AI, designed to run inside your firm.';
      if (intro) intro.textContent = 'Use AI on legal work while keeping sensitive case material under the firm’s control. Lylo is being designed so private case files can remain on the firm’s own systems.';
    }

    let privateCta = document.querySelector('.lylo-private-cta');
    if (!privateCta) {
      privateCta = document.createElement('section');
      privateCta.className = 'lylo-private-cta';
      privateCta.innerHTML = '<div class="lylo-private-cta-inner"><h3>Could this work inside your firm?</h3><p>In a 20-minute call, we will show the closest demonstration, discuss one time-consuming process and explain what a safe evaluation could involve.</p><a class="cta" href="' + bookingHref + '">' + bookingLabel + '</a><span>No preparation, client data or commitment required.</span></div>';
    }
    if (privacy) privacy.insertAdjacentElement('afterend', privateCta);

    const reg = document.querySelector('.reg-trust');
    let regStage = document.querySelector('.lylo-regulatory-stage');
    if (reg && !regStage) {
      regStage = document.createElement('section');
      regStage.className = 'demo-intro lylo-regulatory-stage';
      const inner = document.createElement('div');
      inner.className = 'reveal in';
      regStage.appendChild(inner);
      inner.appendChild(reg);
    }
    if (reg) {
      reg.innerHTML = [
        '<h2 class="reg-question">AI that supports professional judgement.</h2>',
        '<p class="reg-lead">Lylo is being designed around three principles that matter when AI is used in legal work.</p>',
        '<div class="lylo-guidance-context"><span>Developed with current professional guidance in view</span><div class="lylo-guidance-links"><a href="https://lawscot.org.uk/media/pl1lnu5n/ai-guide.pdf" target="_blank" rel="noopener">Law Society of Scotland · Guide to Generative AI</a><a href="https://media.sra.org.uk/solicitors/guidance/misuse-ai/" target="_blank" rel="noopener">England &amp; Wales · 17 Aug 2026 · SRA Misuse of AI warning</a></div></div>',
        '<div class="reg-cards lylo-reg-principles" aria-label="Three principles guiding Lylo">',
          '<article class="reg-card"><strong>Keep sensitive work controlled</strong><span>Lylo is being designed so private legal work can remain within the firm’s environment.</span></article>',
          '<article class="reg-card"><strong>Show the evidence</strong><span>Answers and drafts can be checked against their source documents.</span></article>',
          '<article class="reg-card"><strong>Keep the solicitor in control</strong><span>Outputs remain drafts for professional review, editing and approval.</span></article>',
        '</div>',
        '<details class="lylo-duty-details"><summary>Read how Lylo is being designed around professional duties</summary><div><p>Accuracy, confidentiality, human oversight and professional responsibility remain central. UK GDPR duties—including privacy by design, data minimisation and controlled access—still depend on each firm’s deployment, policies and use.</p><p><a href="https://lawscot.org.uk/media/pl1lnu5n/ai-guide.pdf" target="_blank" rel="noopener">Law Society of Scotland: Guide to Generative AI</a> · <a href="https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/accountability-and-governance/guide-to-accountability-and-governance/data-protection-by-design-and-by-default/" target="_blank" rel="noopener">ICO: data protection by design</a> · <a href="https://media.sra.org.uk/solicitors/guidance/misuse-ai/" target="_blank" rel="noopener">SRA: misuse of AI warning</a></p></div></details>'
      ].join('');
    }
    if (privateCta && regStage) privateCta.insertAdjacentElement('afterend', regStage);

    const beyond = document.querySelector('.beyond-casework');
    if (beyond) {
      const kicker = beyond.querySelector('.beyond-kicker');
      const title = beyond.querySelector('h3');
      const copy = beyond.querySelector('p');
      if (kicker) kicker.textContent = 'Beyond casework';
      if (title) title.textContent = 'A smaller example: out-of-hours calls.';
      if (copy) copy.textContent = 'Lylo can also be adapted for useful firm-wide tasks. The receptionist demonstration shows one possible use outside private legal document work.';
    }
    const phoneAnchor = regStage || privateCta || privacy || mission || other || lastPrincipal;
    if (phoneAnchor && beyond) phoneAnchor.insertAdjacentElement('afterend', beyond);
    if (beyond && phoneSection) beyond.insertAdjacentElement('afterend', phoneSection);
    if (phoneSection) phoneSection.classList.add('lylo-secondary-demo');

    const final = document.querySelector('.final .reveal');
    if (final) {
      const heading = final.querySelector('h3');
      const copy = final.querySelector('p');
      const cta = final.querySelector('.cta');
      if (heading) heading.textContent = 'Start with a focused 20-minute call.';
      if (copy) copy.textContent = 'Tell us where your firm loses time, see the closest working demonstration and decide whether a safe, small evaluation is worth discussing.';
      if (cta) {
        cta.href = bookingHref;
        cta.textContent = bookingLabel;
      }
      let brief = final.querySelector('.lylo-pilot-brief');
      if (!brief) {
        brief = document.createElement('div');
        brief.className = 'lylo-pilot-brief';
        if (cta) cta.insertAdjacentElement('beforebegin', brief);
      }
      brief.innerHTML = [
        '<div class="lylo-pilot-step"><span class="lylo-pilot-step-no">01</span><strong>Tell us where time is lost</strong><span>Choose one legal process that feels repetitive or slow.</span></div>',
        '<div class="lylo-pilot-step"><span class="lylo-pilot-step-no">02</span><strong>See the closest demonstration</strong><span>We will focus the call on the example most relevant to your firm.</span></div>',
        '<div class="lylo-pilot-step"><span class="lylo-pilot-step-no">03</span><strong>Decide if a safe test makes sense</strong><span>If there is no useful fit, nothing further happens.</span></div>'
      ].join('');
    }
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', run, {once:true});
  else run();
})();
