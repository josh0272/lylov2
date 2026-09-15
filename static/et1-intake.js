(() => {
  const form = document.getElementById('et1-form-suggest');
  const input = document.getElementById('et1-form-input');
  const status = document.getElementById('et1-form-status');
  const jump = document.getElementById('et1-suggest-jump');

  if (!form || !input || !status) return;

  const focusInput = () => {
    input.focus({ preventScroll: true });
    input.scrollIntoView({ behavior: 'smooth', block: 'center' });
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
})();