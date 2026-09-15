(() => {
  const findEt1Section = () => {
    return Array.from(document.querySelectorAll('.demo-section')).find((section) => {
      const heading = section.querySelector('.demo-copy h3');
      return heading?.textContent?.trim() === 'Turn case files into a completed form.';
    });
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

    let extension = section.querySelector('.et1-extension');
    if (!extension) {
      extension = buildExtension();
      section.appendChild(extension);
    }

    const existingCta = section.querySelector('.demo-copy .demo-cta');
    if (existingCta && existingCta.id !== 'et1-suggest-jump') {
      const button = document.createElement('button');
      button.type = 'button';
      button.id = 'et1-suggest-jump';
      button.className = 'demo-cta et1-suggest-jump';
      button.textContent = 'Suggest a form';
      existingCta.replaceWith(button);
    }

    const form = document.getElementById('et1-form-suggest');
    const input = document.getElementById('et1-form-input');
    const status = document.getElementById('et1-form-status');
    const jump = document.getElementById('et1-suggest-jump');
    if (!form || !input || !status) return;

    const focusInput = () => {
      input.scrollIntoView({ behavior: 'smooth', block: 'center' });
      window.setTimeout(() => input.focus({ preventScroll: true }), 320);
    };

    jump?.addEventListener('click', (event) => {
      event.preventDefault();
      focusInput();
    });

    form.addEventListener('submit', async (event) => {
      event.preventDefault();

      const value = input.value.trim();
      if (!value) {
        status.className = 'et1-suggest-status error';
        status.textContent = 'Add a form or workflow first.';
        input.focus();
        return;
      }

      const button = form.querySelector('button[type="submit"]');
      button.disabled = true;
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

        status.className = 'et1-suggest-status success';
        status.textContent = 'Thanks. This helps decide what Lylo builds next.';
        input.value = '';
      } catch (_) {
        status.className = 'et1-suggest-status error';
        status.textContent = 'Could not send that just now. Please try again.';
      } finally {
        button.disabled = false;
      }
    });
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init, { once: true });
  } else {
    init();
  }
})();