(() => {
  const run = () => {
    const heroCta = document.querySelector('.hero .cta');
    if (heroCta && !document.querySelector('[data-lylo-demo-reassurance]')) {
      const reassurance = document.createElement('div');
      reassurance.className = 'note';
      reassurance.setAttribute('data-lylo-demo-reassurance', '');
      reassurance.textContent = 'No preparation. No client data. No commitment. See the demos, ask questions and decide whether Lylo is worth exploring further.';

      const demoExplanation = document.createElement('div');
      demoExplanation.className = 'note';
      demoExplanation.setAttribute('data-lylo-demo-explanation', '');
      demoExplanation.textContent = 'In the demo: see the document Q&A, ET1 and Schedule of Loss workflows, hear how the private/on-premise system is being designed, and tell us where it would—or wouldn’t—fit your firm.';

      heroCta.insertAdjacentElement('afterend', reassurance);
      reassurance.insertAdjacentElement('afterend', demoExplanation);
    }

    const final = document.querySelector('.final .reveal');
    if (final && !final.querySelector('[data-lylo-pilot-sequence]')) {
      const sequence = document.createElement('p');
      sequence.setAttribute('data-lylo-pilot-sequence', '');
      sequence.innerHTML = '1. See Lylo in a 20-minute demo.<br>2. If it looks useful, test one workflow with synthetic or properly anonymised material.<br>3. Only discuss a paid pilot if the experiment works.';

      const cta = final.querySelector('.cta');
      if (cta) cta.insertAdjacentElement('beforebegin', sequence);
      else final.appendChild(sequence);
    }
  };

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', run, { once: true });
  else run();
})();