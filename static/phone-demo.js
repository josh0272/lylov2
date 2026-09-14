(() => {
  const demo = document.getElementById('phoneAudioDemo');
  const transcript = document.getElementById('phoneTranscript');
  const wave = document.getElementById('phoneWave');
  const oldButton = document.getElementById('phonePlay');
  const oldAudio = document.getElementById('phoneDemoAudio');

  if (!demo || !transcript || !wave || !oldButton || !oldAudio) return;

  // Replace the old controls so the inline preview script can no longer drive
  // the phone demo with its short placeholder transcript.
  const button = oldButton.cloneNode(true);
  oldButton.replaceWith(button);

  const audio = document.createElement('audio');
  audio.id = 'phoneDemoAudio';
  audio.preload = 'metadata';
  audio.src = '/static/ai-receptionist.mp3';
  oldAudio.replaceWith(audio);

  const icon = button.querySelector('svg');

  if (!document.getElementById('lylo-phone-demo-styles')) {
    const style = document.createElement('style');
    style.id = 'lylo-phone-demo-styles';
    style.textContent = `
      .phone-section .transcript {
        height: 210px !important;
        max-height: 210px !important;
        overflow: hidden !important;
        display: block !important;
        padding: 7px 4px 12px !important;
        scroll-behavior: auto !important;
        overscroll-behavior: contain;
        scrollbar-width: none;
        -ms-overflow-style: none;
        -webkit-mask-image: linear-gradient(to bottom, transparent 0, #000 10px, #000 calc(100% - 12px), transparent 100%);
        mask-image: linear-gradient(to bottom, transparent 0, #000 10px, #000 calc(100% - 12px), transparent 100%);
      }

      .phone-section .transcript::-webkit-scrollbar {
        display: none;
      }

      .phone-section .transcript-line {
        display: block !important;
        width: fit-content !important;
        max-width: 88% !important;
        margin: 0 0 9px !important;
        opacity: 1 !important;
        transform: none !important;
        white-space: normal !important;
        overflow-wrap: anywhere;
      }

      .phone-section .transcript-line.lylo {
        margin-right: auto !important;
        margin-left: 4% !important;
        transform: none !important;
      }

      .phone-section .transcript-line.caller {
        margin-left: auto !important;
        margin-right: 4% !important;
        transform: none !important;
      }

      .phone-section .transcript-speaker {
        display: block;
        margin-bottom: 4px;
        color: #74869d;
        font-size: 9px;
        font-weight: 700;
        letter-spacing: .08em;
        text-transform: uppercase;
      }

      .phone-section .transcript-copy {
        display: block;
      }

      .phone-section .transcript-line.active {
        border-color: rgba(151,177,211,.18) !important;
        background: rgba(121,154,195,.055) !important;
      }

      @media (max-width: 979px) {
        .phone-section .transcript {
          height: 190px !important;
          max-height: 190px !important;
        }
        .phone-section .transcript-line {
          max-width: 90% !important;
        }
      }
    `;
    document.head.appendChild(style);
  }

  // Timings below are aligned to the attached 2:15 recording itself rather
  // than the later Vapi event timestamps. Starts were checked against the
  // recorded speech waveform so the transcript changes with the actual audio.
  const turns = [
    { who: 'lylo', speaker: 'Lylo', start: 1.75, end: 2.20, text: 'Hello.' },
    { who: 'caller', speaker: 'Caller', start: 2.50, end: 3.00, text: 'Hello?' },
    { who: 'lylo', speaker: 'Lylo', start: 3.05, end: 6.45, text: 'You can ask me about Lylo or try the out of hours receptionist demo.' },
    { who: 'caller', speaker: 'Caller', start: 11.30, end: 12.10, text: 'This is good.' },
    { who: 'lylo', speaker: 'Lylo', start: 14.30, end: 20.00, text: 'Good. Lylo is designed to make legal AI feel practical, private, and straightforward.' },
    { who: 'caller', speaker: 'Caller', start: 21.30, end: 26.90, text: 'How is it private? What does it even mean to be on premise? Can you tell me in simple terms?' },
    { who: 'lylo', speaker: 'Lylo', start: 28.60, end: 84.40, text: "Lylo is being designed so that sensitive case documents and the core legal AI can operate within the law firm's own controlled environment. In simple terms, on premises means the system runs on computers and rather than sending case files into a general public AI service. That gives the firm greater control over where information is stored, who can access it, how long it is kept, and whether it can leave the firm's environment. That matters for client confidentiality, legal professional privilege, and data protection. Some optional functions such as telephony can involve carefully selected external providers. Those are separate data flows and must be assessed and configured properly for each firm. Lylo is being built around controlled handling of legal information, source traceability, and solicitor review, not casual uploading of sensitive documents into public AI tools." },
    { who: 'caller', speaker: 'Caller', start: 86.70, end: 87.00, text: 'Cool.' },
    { who: 'lylo', speaker: 'Lylo', start: 89.10, end: 92.40, text: 'Lylo keeps the firm and control of sensitive legal information. Would you like to try the out of hours receptionist demo?' },
    { who: 'caller', speaker: 'Caller', start: 93.00, end: 99.90, text: 'And actually, I would like to ask, can you tell me a joke?' },
    { who: 'lylo', speaker: 'Lylo', start: 101.80, end: 104.10, text: 'Why did the solicitor bring a ladder to court? To reach the higher authority.' },
    { who: 'caller', speaker: 'Caller', start: 105.00, end: 110.80, text: "Why? That's not really that funny. We do have everyone." },
    { who: 'lylo', speaker: 'Lylo', start: 113.00, end: 120.30, text: 'Fair point. That joke should be struck from the record. Why did the lawyer wear 2 pairs of trousers in case they lost their case?' },
    { who: 'caller', speaker: 'Caller', start: 124.00, end: 124.80, text: 'Okay. Good 1.' },
    { who: 'lylo', speaker: 'Lylo', start: 127.20, end: 128.20, text: "I'll take that win." },
    { who: 'caller', speaker: 'Caller', start: 130.90, end: 131.60, text: 'Can you end the call?' },
    { who: 'lylo', speaker: 'Lylo', start: 133.50, end: 134.10, text: 'Goodbye.' }
  ];

  let frame = 0;
  let lastActiveIndex = -1;
  const lineNodes = new Map();

  const setIcon = (playing) => {
    if (!icon) return;
    icon.innerHTML = playing
      ? '<path d="M7 5h4v14H7zm6 0h4v14h-4z"/>'
      : '<path d="M8 5.5v13l10-6.5z"/>';
    button.setAttribute('aria-label', playing ? 'Pause Lylo voice demo' : 'Play Lylo voice demo');
  };

  const resetTranscript = () => {
    transcript.innerHTML = '';
    transcript.scrollTop = 0;
    lineNodes.clear();
    lastActiveIndex = -1;
  };

  const ensureLine = (index) => {
    if (lineNodes.has(index)) return lineNodes.get(index);
    const turn = turns[index];
    const line = document.createElement('div');
    line.className = `transcript-line ${turn.who}`;

    const speaker = document.createElement('span');
    speaker.className = 'transcript-speaker';
    speaker.textContent = turn.speaker;

    const copy = document.createElement('span');
    copy.className = 'transcript-copy';

    line.append(speaker, copy);
    transcript.appendChild(line);
    lineNodes.set(index, { line, copy });
    return { line, copy };
  };

  const scrollActiveIntoView = (node) => {
    if (!node) return;
    const bottom = node.offsetTop + node.offsetHeight;
    const target = Math.max(0, bottom - transcript.clientHeight + 16);
    transcript.scrollTop = target;
  };

  const renderTranscript = (time) => {
    let activeIndex = -1;

    turns.forEach((turn, index) => {
      if (time < turn.start) return;
      const { line, copy } = ensureLine(index);

      const duration = Math.max(0.25, turn.end - turn.start);
      const progress = Math.max(0, Math.min(1, (time - turn.start) / duration));
      const visibleChars = time >= turn.end
        ? turn.text.length
        : Math.max(1, Math.floor(turn.text.length * progress));

      copy.textContent = turn.text.slice(0, visibleChars);

      if (time >= turn.start && time < turn.end) activeIndex = index;
      line.classList.toggle('active', time >= turn.start && time < turn.end);
    });

    if (activeIndex < 0) {
      for (let i = turns.length - 1; i >= 0; i--) {
        if (time >= turns[i].start) {
          activeIndex = i;
          break;
        }
      }
    }

    const active = lineNodes.get(activeIndex)?.line;
    if (active) {
      // Scroll continuously while a long turn grows, and jump naturally to
      // the next speaker when their turn begins.
      scrollActiveIntoView(active);
      lastActiveIndex = activeIndex;
    }
  };

  const updateWave = (time) => {
    const bars = Array.from(wave.children);
    const duration = Number.isFinite(audio.duration) && audio.duration > 0 ? audio.duration : 135.08;
    const progress = Math.max(0, Math.min(1, time / duration));
    bars.forEach((bar, index) => {
      bar.classList.toggle('active', index / Math.max(1, bars.length - 1) <= progress);
    });
  };

  const tick = () => {
    const time = audio.currentTime || 0;
    renderTranscript(time);
    updateWave(time);
    if (!audio.paused && !audio.ended) frame = requestAnimationFrame(tick);
  };

  const startVisuals = () => {
    demo.classList.add('started', 'playing');
    setIcon(true);
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(tick);
  };

  const stopVisuals = () => {
    demo.classList.remove('playing');
    setIcon(false);
    cancelAnimationFrame(frame);
    renderTranscript(audio.currentTime || 0);
    updateWave(audio.currentTime || 0);
  };

  resetTranscript();
  setIcon(false);
  updateWave(0);

  button.addEventListener('click', async () => {
    if (!audio.paused) {
      audio.pause();
      return;
    }

    if (audio.ended || audio.currentTime >= (audio.duration || 135.08) - 0.15) {
      audio.currentTime = 0;
      resetTranscript();
      updateWave(0);
    }

    try {
      await audio.play();
    } catch (error) {
      console.error('Could not play phone demo audio:', error);
    }
  });

  audio.addEventListener('play', startVisuals);
  audio.addEventListener('pause', () => {
    if (!audio.ended) stopVisuals();
  });
  audio.addEventListener('seeking', () => renderTranscript(audio.currentTime || 0));
  audio.addEventListener('ended', () => {
    stopVisuals();
    renderTranscript(audio.duration || 135.08);
    updateWave(audio.duration || 135.08);
  });
})();