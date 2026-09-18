(() => {
  const run = () => {
    const bookingHref = '/call#book';
    const pilotHref = '/call';

    const hero = document.querySelector('.hero');
    const heroCta = hero?.querySelector('.cta');
    if (hero && heroCta) {
      heroCta.textContent = 'Book a 20 minute call';
      heroCta.setAttribute('href', bookingHref);

      const note = hero.querySelector('.note');
      if (note) {
        note.textContent = 'No preparation. No client data. No commitment. See the demos, ask questions and decide whether Lylo is worth exploring further.';
      }

      const subheading = hero.querySelector('.bullet');
      if (subheading) {
        subheading.textContent = 'We are developing private AI for legal work, so solicitors can use the benefits of AI without sending confidential client data to public AI services such as ChatGPT.';
      }
    }

    const desktopNav = document.querySelector('.desktop-nav');
    if (desktopNav && !desktopNav.querySelector('a[href="/research"]')) {
      const researchLink = document.createElement('a');
      researchLink.href = '/research';
      researchLink.textContent = 'Research';
      const bookingButton = desktopNav.querySelector('.desktop-pilot-link');
      if (bookingButton) desktopNav.insertBefore(researchLink, bookingButton);
      else desktopNav.appendChild(researchLink);
    }
    if (desktopNav) {
      const bookingButton = desktopNav.querySelector('.desktop-pilot-link');
      if (bookingButton) {
        bookingButton.textContent = 'Book a 20 minute call';
        bookingButton.setAttribute('href', bookingHref);
      }

      let pilotTextLink = desktopNav.querySelector('.lylo-pilot-nav-text');
      if (!pilotTextLink) {
        pilotTextLink = document.createElement('a');
        pilotTextLink.className = 'lylo-pilot-nav-text';
        pilotTextLink.href = pilotHref;
        pilotTextLink.textContent = 'What happens next';
        if (bookingButton) desktopNav.insertBefore(pilotTextLink, bookingButton);
        else desktopNav.appendChild(pilotTextLink);
      }

      const ordered = [
        desktopNav.querySelector('a[href="#demos"]'),
        desktopNav.querySelector('a[href="#privacy"]'),
        desktopNav.querySelector('a[href="#ai-duties"]'),
        desktopNav.querySelector('a[href="/research"]'),
        pilotTextLink,
        bookingButton
      ];
      ordered.forEach((link) => { if (link) desktopNav.appendChild(link); });
    }

    const mobilePanel = document.querySelector('#mobileMenu .panel-inner');
    if (mobilePanel && !mobilePanel.querySelector('a[href="/research"]')) {
      const researchLink = document.createElement('a');
      researchLink.href = '/research';
      researchLink.textContent = 'Research';
      mobilePanel.appendChild(researchLink);
    }
    if (mobilePanel) {
      let pilotTextLink = mobilePanel.querySelector('.lylo-pilot-nav-text');
      if (!pilotTextLink) {
        const existingPilot = Array.from(mobilePanel.querySelectorAll('a')).find((link) => link.getAttribute('href') === pilotHref);
        if (existingPilot) {
          pilotTextLink = existingPilot;
          pilotTextLink.classList.add('lylo-pilot-nav-text');
          pilotTextLink.textContent = 'What happens next';
        } else {
          pilotTextLink = document.createElement('a');
          pilotTextLink.className = 'lylo-pilot-nav-text';
          pilotTextLink.href = pilotHref;
          pilotTextLink.textContent = 'What happens next';
        }
      }

      let bookingLink = mobilePanel.querySelector('.lylo-mobile-booking-link');
      if (!bookingLink) {
        bookingLink = document.createElement('a');
        bookingLink.className = 'lylo-mobile-booking-link';
        bookingLink.href = bookingHref;
        bookingLink.textContent = 'Book a 20 minute call';
      }

      const ordered = [
        mobilePanel.querySelector('a[href="#demos"]'),
        mobilePanel.querySelector('a[href="#privacy"]'),
        mobilePanel.querySelector('a[href="#ai-duties"]'),
        mobilePanel.querySelector('a[href="/research"]'),
        pilotTextLink,
        bookingLink
      ];
      ordered.forEach((link) => { if (link) mobilePanel.appendChild(link); });
    }

    document.querySelector('.lylo-research-strip')?.remove();

    if (window.innerWidth >= 980 && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      const oldCarousel = document.querySelector('.reg-cards');
      if (oldCarousel && !oldCarousel.classList.contains('lylo-smooth-carousel')) {
        const carousel = oldCarousel.cloneNode(true);
        carousel.querySelectorAll('[aria-hidden="true"]').forEach((clone) => clone.remove());
        carousel.classList.add('lylo-smooth-carousel');
        oldCarousel.replaceWith(carousel);

        const originals = Array.from(carousel.children);
        originals.forEach((card) => {
          const clone = card.cloneNode(true);
          clone.setAttribute('aria-hidden', 'true');
          clone.setAttribute('tabindex', '-1');
          carousel.appendChild(clone);
        });

        const track = document.createElement('div');
        track.className = 'lylo-reg-track';
        while (carousel.firstChild) track.appendChild(carousel.firstChild);
        carousel.appendChild(track);

        const firstClone = track.children[originals.length];
        let offset = 0;
        let lastTime = performance.now();
        let interactionUntil = 0;
        let dragging = false;
        let dragStartX = 0;
        let dragStartOffset = 0;

        const loopWidth = () => firstClone ? firstClone.offsetLeft - track.children[0].offsetLeft : 0;
        const normalise = () => {
          const width = loopWidth();
          if (!width) return;
          while (offset >= width) offset -= width;
          while (offset < 0) offset += width;
        };
        const render = () => {
          normalise();
          track.style.transform = `translate3d(${-offset}px,0,0)`;
        };
        const pause = (ms = 750) => { interactionUntil = performance.now() + ms; };

        carousel.addEventListener('wheel', (event) => {
          const delta = Math.abs(event.deltaY) >= Math.abs(event.deltaX) ? event.deltaY : event.deltaX;
          if (!delta) return;
          event.preventDefault();
          offset += delta;
          pause(900);
          render();
        }, { passive: false });

        carousel.addEventListener('pointerdown', (event) => {
          if (event.pointerType !== 'mouse' || event.button !== 0) return;
          dragging = true;
          dragStartX = event.clientX;
          dragStartOffset = offset;
          carousel.classList.add('dragging');
          carousel.setPointerCapture?.(event.pointerId);
          pause(5000);
        });
        carousel.addEventListener('pointermove', (event) => {
          if (!dragging) return;
          offset = dragStartOffset - (event.clientX - dragStartX);
          pause(5000);
          render();
        });
        const endDrag = (event) => {
          if (!dragging) return;
          dragging = false;
          carousel.classList.remove('dragging');
          try { carousel.releasePointerCapture?.(event.pointerId); } catch (_) {}
          pause(800);
        };
        carousel.addEventListener('pointerup', endDrag);
        carousel.addEventListener('pointercancel', endDrag);

        const animate = (now) => {
          const dt = Math.min(now - lastTime, 24);
          if (!dragging && now > interactionUntil) {
            offset += dt * 0.032;
            render();
          }
          lastTime = now;
          requestAnimationFrame(animate);
        };
        render();
        requestAnimationFrame(animate);
      }
    }

    const demoIntro = document.querySelector('.demo-intro');

    if (demoIntro && !document.querySelector('.lylo-mission')) {
      const mission = document.createElement('section');
      mission.className = 'lylo-mission';
      mission.innerHTML = `
        <div class="lylo-mission-inner">
          <div class="lylo-mission-kicker">Our mission</div>
          <h2>To develop private, firm-controlled AI that is practical and accessible to every law firm, regardless of size.</h2>
          <p>So solicitors can benefit from AI while keeping confidential client information under their firm’s control.</p>
        </div>`;
      demoIntro.insertAdjacentElement('beforebegin', mission);
    }

    const reg = document.querySelector('.reg-trust');
    const privacy = document.querySelector('.privacy');
    const demoSections = Array.from(document.querySelectorAll('.demo-section'));
    const phoneSection = demoSections.find((section) => section.classList.contains('phone-section'));
    const legalSections = demoSections.filter((section) => !section.classList.contains('phone-section'));

    if (demoIntro) demoIntro.classList.add('lylo-demos-first');

    let regStage = document.querySelector('.lylo-regulatory-stage');
    if (reg && !regStage) {
      regStage = document.createElement('section');
      regStage.className = 'demo-intro lylo-regulatory-stage';
      const inner = document.createElement('div');
      inner.className = 'reveal in';
      regStage.appendChild(inner);
      inner.appendChild(reg);
    }

    const lastLegal = legalSections[legalSections.length - 1];
    if (lastLegal && privacy) lastLegal.insertAdjacentElement('afterend', privacy);
    if (privacy && regStage) privacy.insertAdjacentElement('afterend', regStage);

    let midBooking = document.querySelector('.lylo-mid-booking');
    if (regStage && !midBooking) {
      midBooking = document.createElement('section');
      midBooking.className = 'lylo-mid-booking';
      midBooking.innerHTML = `<a class="cta" href="${bookingHref}">Book a 20 minute call</a><p>No preparation. No client data. No commitment.</p>`;
      regStage.insertAdjacentElement('afterend', midBooking);
    }

    const beyond = document.querySelector('.beyond-casework');
    const preBeyondAnchor = midBooking || regStage;
    if (preBeyondAnchor && beyond) preBeyondAnchor.insertAdjacentElement('afterend', beyond);
    if (phoneSection) {
      const anchor = beyond || midBooking || regStage || privacy || lastLegal;
      if (anchor) anchor.insertAdjacentElement('afterend', phoneSection);
    }

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
        link.textContent = 'Book a 20 minute call';
        cta.replaceWith(link);
      } else {
        cta.textContent = 'Book a 20 minute call';
        cta.setAttribute('href', bookingHref);
      }
    });

    const final = document.querySelector('.final .reveal');
    if (final) {
      const finalCopy = final.querySelector('p');
      if (finalCopy) {
        finalCopy.textContent = 'Our proof-of-concept demos show what we can build, but Lylo is not limited to these examples. Start with a 20-minute call to discuss the processes that take time in your firm and explore a private AI workflow built around the way you work.';
      }

      final.querySelector('.lylo-demo-explainer')?.remove();

      const finalCta = final.querySelector('.cta');
      if (finalCta) {
        finalCta.textContent = 'Book a 20 minute call';
        finalCta.setAttribute('href', bookingHref);
      }
    }

    let brief = final?.querySelector('.lylo-pilot-brief');
    if (final && !brief) {
      brief = document.createElement('div');
      brief.className = 'lylo-pilot-brief';
      brief.setAttribute('aria-label', 'Founding pilot steps');
      brief.innerHTML = `
        <div class="lylo-pilot-step">
          <span class="lylo-pilot-step-no">01</span>
          <strong>Start with a 20-minute call with a co founder</strong>
          <span>Tell us which legal processes take time and see the most relevant proof-of-concept demonstrations.</span>
        </div>
        <div class="lylo-pilot-step">
          <span class="lylo-pilot-step-no">02</span>
          <strong>Test one safe workflow</strong>
          <span>If Lylo looks useful, choose one workflow to evaluate using synthetic or properly anonymised material and compare it with your normal process.</span>
        </div>
        <div class="lylo-pilot-step">
          <span class="lylo-pilot-step-no">03</span>
          <strong>Review the evidence</strong>
          <span>Review what worked, what needs improvement and whether Lylo provided real value. Only discuss a paid pilot if the results justify taking it further.</span>
        </div>`;

      const cta = final.querySelector('.cta');
      if (cta) cta.insertAdjacentElement('beforebegin', brief);
      else final.appendChild(brief);
    } else if (brief) {
      const steps = brief.querySelectorAll('.lylo-pilot-step');
      const copy = [
        ['Start with a 20-minute call with a co founder', 'Tell us which legal processes take time and see the most relevant proof-of-concept demonstrations.'],
        ['Test one safe workflow', 'If Lylo looks useful, choose one workflow to evaluate using synthetic or properly anonymised material and compare it with your normal process.'],
        ['Review the evidence', 'Review what worked, what needs improvement and whether Lylo provided real value. Only discuss a paid pilot if the results justify taking it further.']
      ];
      steps.forEach((step, index) => {
        if (!copy[index]) return;
        const heading = step.querySelector('strong');
        const body = step.querySelector('span:last-child');
        if (heading) heading.textContent = copy[index][0];
        if (body) body.textContent = copy[index][1];
      });
    }

    final?.querySelector('.lylo-pilot-more')?.remove();

    
  };

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', run, { once: true });
  else run();
})();