(() => {
  const run = () => {
    const reg = document.querySelector('.reg-trust');
    if (reg) {
      const tag = reg.querySelector('.reg-date');
      if (tag) {
        tag.innerHTML = '<span class="reg-pill">Law Society of Scotland · Guide to Generative AI</span><span class="reg-pill reg-pill-secondary">England & Wales · 17 Aug 2026 · SRA Misuse of AI warning</span>';
        tag.classList.add('reg-date-pills');
      }

      const heading = reg.querySelector('.reg-question');
      if (heading) heading.textContent = 'AI use in legal work is changing.';

      if (heading && !reg.querySelector('.reg-mobile-framework')) {
        const framework = document.createElement('div');
        framework.className = 'reg-mobile-framework';
        framework.innerHTML = '<p>UK GDPR works alongside these professional responsibilities. The wider UK framework also includes the Data Protection Act 2018, the Data (Use and Access) Act 2025, and PECR 2003 where relevant.</p><strong>Lylo is designed with these responsibilities in mind from the beginning.</strong><span>Law Society of Scotland: Guide to Generative AI &nbsp;·&nbsp; ICO: data protection by design &nbsp;·&nbsp; England &amp; Wales: SRA AI warning</span>';
        heading.insertAdjacentElement('afterend', framework);
      }

      const lead = reg.querySelector('.reg-lead');
      if (lead) lead.textContent = 'Legal regulators and professional bodies are increasingly addressing inaccurate AI output, confidentiality, supervision and professional responsibility. Lylo is being designed around those risks.';

      if (lead && !reg.querySelector('.reg-lead-mobile')) {
        const mobileLead = document.createElement('p');
        mobileLead.className = 'reg-lead reg-lead-mobile';
        mobileLead.textContent = 'Regulators and professional bodies are increasingly addressing accuracy, confidentiality, supervision and professional responsibility.';
        lead.insertAdjacentElement('afterend', mobileLead);
      }

      reg.querySelectorAll('.reg-card').forEach((card) => {
        const title = card.querySelector('strong')?.textContent.trim();
        const lylo = card.querySelector('em');
        if (title === 'Accuracy & quality' && lylo) {
          lylo.textContent = 'Lylo is being designed to ground answers in the firm’s own documents and show the source material for review.';
        } else if (title === 'Human oversight' && lylo) {
          lylo.textContent = 'Lylo is being designed to keep the solicitor in control, with checking, editing and review built into the workflow.';
        }
      });

      const line = reg.querySelector('.reg-line');
      if (line) {
        line.innerHTML = 'UK GDPR works alongside these professional responsibilities. The wider UK framework also includes the Data Protection Act 2018, the Data (Use and Access) Act 2025, and PECR 2003 where relevant.<span class="reg-built">Lylo is designed with these responsibilities in mind from the beginning.</span>';
      }

      const cards = reg.querySelector('.reg-cards');
      const framework = reg.querySelector('.reg-mobile-framework');
      if (cards && framework) cards.insertAdjacentElement('afterend', framework);
    }

    const tagline = document.querySelector('.hero .tagline');
    if (tagline) tagline.textContent = 'Private AI, designed for solicitors.';

    const heroSupportingCopy = document.querySelector('.hero .bullet');
    if (heroSupportingCopy) heroSupportingCopy.textContent = 'Retrieval-based AI system in development, with specialist modules being designed to support drafting Statements of Fact, ET1s, Schedules of Loss and Simple Procedure claim forms for solicitor review.';

    const demoHeading = document.querySelector('.demo-heading');
    if (demoHeading) {
      const title = demoHeading.querySelector('h2');
      const copy = demoHeading.querySelector('p');
      if (title) title.textContent = 'See what we’re building.';
      if (copy) copy.textContent = 'Four proof-of-concept demos. Each one shows a real task Lylo is being designed to help with.';
    }

    document.querySelectorAll('.demo-section').forEach((section) => {
      const heading = section.querySelector('.demo-copy h3');
      const headingText = heading?.textContent.trim();
      const copy = section.querySelector('.demo-copy p');
      if (!heading || !copy) return;

      if (headingText === 'Ask the case. Get the answer.') {
        copy.textContent = 'Upload the case files and ask Lylo a question. It is being designed to search the documents, find the key facts and bring the answer back in one place.';
        heading.textContent = 'Ask the case. Find the source.';
      } else if (headingText === 'Turn case files into a completed form.') {
        copy.textContent = 'Lylo is being designed to pull names, dates and case details from uploaded documents, fill the ET1 and prepare the information for review. You stay in control before it is used.';
        heading.textContent = 'Turn case files into a draft form.';
      } else if (headingText === 'Know what the claim is worth.') {
        copy.textContent = 'Lylo’s Schedule of Loss calculator is being designed to use the claimant’s pay, dates and losses to calculate the figure and build a clear schedule for review.';
        heading.textContent = 'Build a Schedule of Loss.';
      }
    });

    const et1ExtensionHead = document.querySelector('.et1-extension-head');
    if (et1ExtensionHead) {
      const extensionCopy = et1ExtensionHead.querySelector('p');
      if (extensionCopy) extensionCopy.textContent = 'The same approach could be extended to other forms your firm uses.';
      if (!et1ExtensionHead.querySelector('.et1-potential-modules-label')) {
        const label = document.createElement('p');
        label.className = 'et1-potential-modules-label';
        label.textContent = 'Potential form modules include:';
        et1ExtensionHead.appendChild(label);
      }
    }

    const privacySection = document.querySelector('.privacy');
    if (privacySection) {
      const heading = privacySection.querySelector('h3');
      if (heading) heading.textContent = 'Private AI, designed to run inside your firm.';

      const intro = privacySection.querySelector('.privacy-main');
      if (intro) intro.textContent = 'What if your firm could use the benefits of AI without sending sensitive client data to a public AI service? Lylo is being designed so case files can stay on the firm’s own systems.';

      privacySection.querySelectorAll('.privacy-node').forEach((node) => {
        const title = node.querySelector('strong')?.textContent.trim();
        const copy = node.querySelector('span');
        if (title === 'On-premise AI' && copy) {
          copy.textContent = 'The private legal AI is being designed to run on hardware inside the firm.';
        } else if (title === 'Local case files' && copy) {
          copy.textContent = 'The on-premise design is intended to keep sensitive documents on the firm’s own network.';
        } else if (title === 'Clear cloud boundary' && copy) {
          copy.textContent = 'The private legal AI is being designed to keep legal work local, while phone and voice services remain separate.';
        }
      });

      const note = privacySection.querySelector('.privacy-note');
      if (note) note.textContent = 'Lylo is being designed around UK GDPR principles including privacy by design, data minimisation and controlled access. Final compliance still depends on each firm’s deployment, policies and use.';
    }

    const heroCta = document.querySelector('.hero .cta');
    if (heroCta) heroCta.setAttribute('href', '/founding-pilot');

    const people = document.querySelector('.lylo-people');
    if (people) {
      const kicker = people.querySelector('.lylo-section-kicker');
      if (kicker) kicker.textContent = 'Meet the co-founders of Lylo';

      const intro = people.querySelector('.lylo-people-intro');
      if (intro) intro.textContent = 'A small team combining engineering and legal backgrounds while Lylo is being researched, tested and shaped.';

      const link = people.querySelector('.lylo-secondary-link');
      if (link) link.innerHTML = 'Meet the people behind Lylo <span class="arrow">→</span>';

      people.querySelectorAll('.lylo-person').forEach((person) => {
        const name = person.querySelector('strong')?.textContent.trim();
        const credential = person.querySelector('.credential');
        if (name === 'Jessica Jayan' && credential) {
          credential.textContent = 'Scots (Clinical) LLB · University of Strathclyde';
        }
      });
    }

    const research = document.querySelector('.lylo-research-strip');
    if (research) {
      const heading = research.querySelector('h3');
      const body = research.querySelector('p');
      const note = research.querySelector('.lylo-research-note');
      if (heading) heading.textContent = 'Help us develop Lylo around real legal work.';
      if (body) body.textContent = 'We are speaking with people in law about the work that takes the most time, where AI could genuinely help, and what firms would need before trusting it. Our short questionnaire helps guide Lylo’s development.';
      if (note) note.innerHTML = 'Around 5 minutes · Used for product research · We use responses for product research. See our <a href="/static/privacy.html">Privacy notice</a> for how personal data is collected, used and retained.';
    }

    const final = document.querySelector('.final .reveal');
    if (final) {
      const copy = final.querySelector('p');
      const cta = final.querySelector('.cta');
      if (copy) copy.textContent = 'The demos shown above are proof-of-concept demonstrations. Start with a short demo. If it looks useful, test Lylo on one synthetic or properly anonymised matter and compare it with your normal workflow.';
      if (cta) cta.setAttribute('href', '/founding-pilot');
    }

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
        .reg-lead-mobile,.reg-mobile-framework{display:none}
        .lylo-research-note a{color:inherit!important;text-decoration:underline;text-decoration-color:currentColor;text-underline-offset:2px}
        @media(max-width:700px){
          .reg-date-pills{gap:6px}.reg-pill{min-height:26px;padding:0 9px;font-size:9px;text-align:center}
          .reg-trust>.reg-lead:not(.reg-lead-mobile){display:none}
          .reg-lead-mobile{display:block;margin:13px auto 20px}
          .reg-trust>.reg-line,.reg-trust>.reg-sources{display:none}
          .reg-mobile-framework{display:block;margin:24px auto 18px;max-width:700px;padding:18px 0;border-top:1px solid rgba(255,255,255,.07);border-bottom:1px solid rgba(255,255,255,.07);color:#91a0b3;text-align:center}
          .reg-mobile-framework p{margin:0;font-size:11px;line-height:1.55}
          .reg-mobile-framework strong{display:block;margin-top:9px;color:#dce6f2;font-size:11.5px;line-height:1.5;font-weight:600}
          .reg-mobile-framework span{display:block;margin-top:10px;padding-top:10px;border-top:1px solid rgba(255,255,255,.055);color:#73859b;font-size:9.5px;line-height:1.55}
        }
      `;
      document.head.appendChild(style);
    }
  };

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', run, { once: true });
  else run();
})();