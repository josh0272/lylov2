(() => {
  const findEt1Section = () => {
    return Array.from(document.querySelectorAll('.demo-section')).find((section) => {
      const heading = section.querySelector('.demo-copy h3');
      return heading?.textContent?.trim() === 'Turn case files into a completed form.';
    });
  };

  const initEt1Captions = (section) => {
    const video = section.querySelector('.demo-media video');
    if (!video || video.dataset.lyloEt1Captions === '1') return;
    video.dataset.lyloEt1Captions = '1';

    const track = document.createElement('track');
    track.kind = 'captions';
    track.label = 'English';
    track.srclang = 'en';
    track.src = '/static/et1-en.vtt';
    track.default = true;
    track.dataset.lyloCaptionTrack = 'et1';
    video.appendChild(track);

    const media = video.closest('.demo-media');
    if (!media || media.querySelector('.lylo-cc-toggle')) return;

    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'lylo-cc-toggle';
    button.textContent = 'CC';
    button.setAttribute('aria-label', 'Hide captions');
    button.setAttribute('aria-pressed', 'true');
    media.appendChild(button);

    let captionsOn = true;

    const getTextTrack = () => track.track || Array.from(video.textTracks || []).find((item) => item.language === 'en');
    const applyCaptionState = () => {
      const textTrack = getTextTrack();
      if (textTrack) textTrack.mode = captionsOn ? 'showing' : 'hidden';
      button.setAttribute('aria-pressed', captionsOn ? 'true' : 'false');
      button.setAttribute('aria-label', captionsOn ? 'Hide captions' : 'Show captions');
      button.title = captionsOn ? 'Captions on' : 'Captions off';
    };

    button.addEventListener('click', (event) => {
      event.preventDefault();
      event.stopPropagation();
      captionsOn = !captionsOn;
      applyCaptionState();
    });

    track.addEventListener('load', applyCaptionState);
    video.addEventListener('loadedmetadata', applyCaptionState, { once: true });
    window.setTimeout(applyCaptionState, 250);
  };

  const buildExtension = () => {
    const wrap = document.createElement('div');
    wrap.className = 'et1-extension reveal in';
    wrap.innerHTML = `
      <div class="et1-extension-head">
        <h4>One form is just the start.</h4>
        <p>Lylo can be built to fill the forms your firm uses every day.</p>
      </div>
      <div class="et1-form-strip" aria-label="Examples of forms Lylo could be adapted to fill">
        <div class="et1-form-track">
          <div class="et1-form-set">
            <span class="et1-form-chip">ET1</span>
            <span class="et1-form-chip">ET3</span>
            <span class="et1-form-chip">N1 Claim Form</span>
            <span class="et1-form-chip">N244 Application</span>
            <span class="et1-form-chip">C100 Family Application</span>
            <span class="et1-form-chip">Form E</span>
            <span class="et1-form-chip">Simple Procedure</span>
            <span class="et1-form-chip is-own">Your firm’s own forms</span>
          </div>
          <div class="et1-form-set" aria-hidden="true">
            <span class="et1-form-chip">ET1</span>
            <span class="et1-form-chip">ET3</span>
            <span class="et1-form-chip">N1 Claim Form</span>
            <span class="et1-form-chip">N244 Application</span>
            <span class="et1-form-chip">C100 Family Application</span>
            <span class="et1-form-chip">Form E</span>
            <span class="et1-form-chip">Simple Procedure</span>
            <span class="et1-form-chip is-own">Your firm’s own forms</span>
          </div>
        </div>
      </div>
      <form class="et1-suggest" id="et1-form-suggest">
        <label for="et1-form-input">What form takes your firm too much time?</label>
        <div class="et1-suggest-row">
          <input id="et1-form-input" name="form_suggestion" type="text" autocomplete="off" maxlength="140" placeholder="e.g. ET3, Form E, our client intake form…" aria-describedby="et1-form-status">
          <button type="submit">Suggest a form</button>
          <a class="et1-book-demo" href="/research">Book a 20-minute demo</a>
        </div>
        <div class="et1-suggest-status" id="et1-form-status" aria-live="polite"></div>
      </form>
    `;
    return wrap;
  };

  const init = () => {
    const section = findEt1Section();
    if (!section) return;

    section.classList.add('et1-section');
    initEt1Captions(section);

    let extension = section.querySelector('.et1-extension');
    if (!extension) {
      extension = buildExtension();
      section.appendChild(extension);
    }

    const existingCta = section.querySelector('.demo-copy .demo-cta');
    if (existingCta) {
      if (existingCta.tagName === 'A') {
        existingCta.href = '/research';
        existingCta.textContent = 'Explore the founding pilot';
      } else {
        const link = document.createElement('a');
        link.href = '/research';
        link.className = 'demo-cta';
        link.textContent = 'Explore the founding pilot';
        existingCta.replaceWith(link);
      }
    }

    const form = document.getElementById('et1-form-suggest');
    const input = document.getElementById('et1-form-input');
    const status = document.getElementById('et1-form-status');
    const submitButton = form?.querySelector('button[type="submit"]');
    const label = form?.querySelector('label');
    if (!form || !input || !status || !submitButton || !label) return;

    const chips = Array.from(section.querySelectorAll('.et1-form-chip'));

    const setReadyState = (ready, selectedValue = '') => {
      if (form.classList.contains('is-submitted')) return;
      submitButton.classList.toggle('is-ready', ready);
      submitButton.textContent = ready ? 'Send suggestion' : 'Suggest a form';

      chips.forEach((chip) => {
        const selected = Boolean(selectedValue) && chip.textContent.trim() === selectedValue;
        chip.classList.toggle('is-selected', selected);
        chip.setAttribute('aria-pressed', selected ? 'true' : 'false');
      });
    };

    const chooseChip = (chip) => {
      if (form.classList.contains('is-submitted')) return;
      const value = chip.textContent.trim();
      input.value = value;
      status.className = 'et1-suggest-status';
      status.textContent = '';
      setReadyState(true, value);

      submitButton.classList.remove('ready-cue');
      void submitButton.offsetWidth;
      submitButton.classList.add('ready-cue');
      window.setTimeout(() => submitButton.classList.remove('ready-cue'), 900);

      form.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    };

    chips.forEach((chip) => {
      const duplicateSet = chip.closest('.et1-form-set')?.getAttribute('aria-hidden') === 'true';
      if (!duplicateSet) {
        chip.setAttribute('role', 'button');
        chip.setAttribute('tabindex', '0');
        chip.setAttribute('aria-pressed', 'false');
      }

      chip.addEventListener('click', () => chooseChip(chip));
      if (!duplicateSet) {
        chip.addEventListener('keydown', (event) => {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            chooseChip(chip);
          }
        });
      }
    });

    input.addEventListener('input', () => {
      const value = input.value.trim();
      const exactChip = chips.find((chip) => chip.textContent.trim() === value);
      setReadyState(Boolean(value), exactChip ? value : '');
      if (status.textContent) {
        status.className = 'et1-suggest-status';
        status.textContent = '';
      }
    });

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      if (form.classList.contains('is-submitted')) return;

      const value = input.value.trim();
      if (!value) {
        status.className = 'et1-suggest-status error';
        status.textContent = 'Add a form or workflow first.';
        input.focus();
        return;
      }

      submitButton.disabled = true;
      status.className = 'et1-suggest-status';
      status.textContent = 'Sending…';

      try {
        const body = new FormData();
        body.append('name', 'Lylo website form suggestion');
        body.append('email', '');
        body.append('answers', `Form or workflow suggestion: ${value}`);
        body.append('transcript', '');

        const response = await fetch('/api/submit', { method: 'POST', body });
        const data = await response.json().catch(() => ({}));
        if (!response.ok || data.ok === false) throw new Error(data.error || 'Could not send');

        form.classList.add('is-submitted');
        label.textContent = 'Thanks — want to see Lylo on your workflow?';
        status.className = 'et1-suggest-status';
        status.textContent = '';
        submitButton.classList.remove('is-ready', 'ready-cue');
        submitButton.textContent = '✓ Suggestion sent';
        submitButton.disabled = true;
        input.value = '';
        chips.forEach((chip) => chip.classList.remove('is-selected'));
      } catch (_) {
        status.className = 'et1-suggest-status error';
        status.textContent = 'Could not send that just now. Please try again.';
        submitButton.disabled = false;
      }
    });
  };

  const loadPreviewCtas = () => {
    if (document.getElementById('preview-cta-script')) return;
    const script = document.createElement('script');
    script.id = 'preview-cta-script';
    script.src = '/static/preview-cta.js?v=2';
    script.defer = true;
    document.body.appendChild(script);
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => { init(); loadPreviewCtas(); }, { once: true });
  } else {
    init();
    loadPreviewCtas();
  }
})();