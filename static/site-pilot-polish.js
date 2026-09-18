(() => {
  const run = () => {
    const bookingHref = '/call#book';
    const pilotHref = '/call';

    const hero = document.querySelector('.hero');
    const heroCta = hero?.querySelector('.cta');
    if (hero && heroCta) {
      heroCta.textContent = 'Book a 20-minute call';
      heroCta.setAttribute('href', bookingHref);

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
      const bookingButton = desktopNav.querySelector('.desktop-pilot-link');
      if (bookingButton) desktopNav.insertBefore(researchLink, bookingButton);
      else desktopNav.appendChild(researchLink);
    }
    if (desktopNav) {
      const bookingButton = desktopNav.querySelector('.desktop-pilot-link');
      if (bookingButton) {
        bookingButton.textContent = 'Book a 20-minute call';
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
        bookingLink.textContent = 'Book a 20-minute call';
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
      midBooking.innerHTML = `<a class="cta" href="${bookingHref}">Book a 20-minute call</a><p>No preparation. No client data. No commitment.</p>`;
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
        link.textContent = 'Book a 20-minute call';
        cta.replaceWith(link);
      } else {
        cta.textContent = 'Book a 20-minute call';
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
        finalCta.textContent = 'Book a 20-minute call';
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
          <strong>Start with a 20-minute call</strong>
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
        ['Start with a 20-minute call', 'Tell us which legal processes take time and see the most relevant proof-of-concept demonstrations.'],
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

    if (!document.getElementById('lylo-pilot-polish-styles')) {
      const style = document.createElement('style');
      style.id = 'lylo-pilot-polish-styles';
      style.textContent = `
        .lylo-demos-first .demo-heading{margin-top:0;padding-top:0;border-top:0}
        .lylo-regulatory-stage{padding-top:105px;padding-bottom:92px}
        .lylo-mid-booking{padding:52px var(--gutter) 68px;text-align:center;border-top:1px solid rgba(255,255,255,.045)}
        .lylo-mid-booking .cta{margin:0 auto 12px}
        .lylo-mid-booking p{margin:0;color:#7e8da1;font-size:12px;line-height:1.5}
        .lylo-pilot-brief{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));width:min(100%,820px);margin:30px auto 20px;border-top:1px solid rgba(255,255,255,.065);border-bottom:1px solid rgba(255,255,255,.065)}
        .lylo-pilot-step{position:relative;padding:19px 24px;text-align:left}
        .lylo-pilot-step+.lylo-pilot-step{border-left:1px solid rgba(255,255,255,.055)}
        .lylo-pilot-step-no{display:block;margin-bottom:8px;color:#7896bb;font-size:9px;font-weight:700;letter-spacing:.13em}
        .lylo-pilot-step strong{display:block;margin-bottom:6px;color:#e7edf5;font-size:12.5px;font-weight:600;line-height:1.35}
        .lylo-pilot-step>span:last-child{display:block;color:#7e8da1;font-size:10.5px;line-height:1.5}
        .lylo-pilot-more{display:block;width:max-content;max-width:100%;margin:0 auto 24px;color:#8fa8c7;text-decoration:none;border-bottom:1px solid rgba(143,168,199,.28);font-size:12px;line-height:1.5}
        .lylo-pilot-more:hover{color:#c7dcf5}
        @media(min-width:980px){
          .final .reveal>p{max-width:820px;font-size:18px;line-height:1.7;margin-bottom:34px}
          .lylo-pilot-brief{width:min(100%,980px);margin:38px auto 22px}
          .lylo-pilot-step{padding:27px 32px}
          .lylo-pilot-step-no{font-size:11px;margin-bottom:10px}
          .lylo-pilot-step strong{font-size:15px;margin-bottom:8px}
          .lylo-pilot-step>span:last-child{font-size:12.5px;line-height:1.58}
          .reg-cards.lylo-smooth-carousel{display:block;overflow:hidden;scroll-behavior:auto;will-change:transform;transform:translateZ(0);backface-visibility:hidden}
          .reg-cards.lylo-smooth-carousel .lylo-reg-track{display:flex;gap:18px;width:max-content;padding:8px 4px 10px;will-change:transform;transform:translate3d(0,0,0);backface-visibility:hidden}
        }
        @media(max-width:700px){
          .lylo-regulatory-stage{padding-top:74px;padding-bottom:64px}
          .lylo-mid-booking{padding:38px 18px 52px}
          .lylo-mid-booking p{font-size:11px}
          .lylo-pilot-brief{grid-template-columns:1fr;width:min(100%,350px);margin:26px auto 18px}
          .lylo-pilot-step{padding:15px 4px;text-align:center}
          .lylo-pilot-step+.lylo-pilot-step{border-left:0;border-top:1px solid rgba(255,255,255,.055)}
          .lylo-pilot-step-no{margin-bottom:6px}
          .lylo-pilot-step strong{font-size:12px}
          .lylo-pilot-step>span:last-child{max-width:300px;margin:0 auto;font-size:10.5px}
          .lylo-pilot-more{margin-bottom:20px;font-size:11px}
        }
      `;
      document.head.appendChild(style);
    }
  };

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', run, { once: true });
  else run();
})();