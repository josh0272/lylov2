(() => {
  const demo = document.getElementById('phoneAudioDemo');
  const stage = demo?.closest('.audio-stage');
  const transcript = document.getElementById('phoneTranscript');
  const wave = document.getElementById('phoneWave');
  const oldButton = document.getElementById('phonePlay');
  const oldAudio = document.getElementById('phoneDemoAudio');

  if (!demo || !stage || !transcript || !wave || !oldButton || !oldAudio) return;
  if (demo.dataset.phoneDemoV2 === '1') return;
  demo.dataset.phoneDemoV2 = '1';

  const button = oldButton.cloneNode(true);
  oldButton.replaceWith(button);

  const audio = document.createElement('audio');
  audio.id = 'phoneDemoAudio';
  audio.preload = 'metadata';
  audio.src = '/static/ai-receptionist.mp3';
  oldAudio.replaceWith(audio);

  const icon = button.querySelector('svg');
  const FALLBACK_DURATION = 135.144;
  const TYPE_CHARS_PER_SECOND = 38;
  let mediaDuration = FALLBACK_DURATION;
  let frame = 0;
  const lineNodes = new Map();

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

      .phone-section .transcript::-webkit-scrollbar { display: none; }

      .phone-section .transcript-line {
        display: block !important;
        width: fit-content !important;
        max-width: 88% !important;
        margin: 0 0 9px !important;
        padding: 9px 13px !important;
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

      .phone-section .transcript-copy { display: block; }

      .phone-section .transcript-line.active {
        border-color: rgba(151,177,211,.18) !important;
        background: rgba(121,154,195,.055) !important;
      }

      @media (max-width: 979px) {
        .phone-section .audio-stage.phone-demo-active {
          height: 445px !important;
        }

        .phone-section .transcript {
          height: 190px !important;
          max-height: 190px !important;
          width: min(94%,430px) !important;
        }

        .phone-section .transcript-line {
          max-width: 90% !important;
        }
      }
    `;
    document.head.appendChild(style);
  }

  const turns = [
    { who:'lylo', speaker:'Lylo', start:2.96, end:6.45, text:'Hello. You can ask me about Lylo or try the out of hours receptionist demo.' },
    { who:'caller', speaker:'Caller', start:12.24, text:'This is good.' },
    { who:'lylo', speaker:'Lylo', start:15.30, text:'Good. Lylo is designed to make legal AI feel practical, private, and straightforward.' },
    { who:'caller', speaker:'Caller', start:22.45, text:'How is it private? What does it even mean to be on premise? Can you tell me in simple terms?' },
    { who:'lylo', speaker:'Lylo', start:29.61, text:"Lylo is being designed so that sensitive case documents and the core legal AI can operate within the law firm's own controlled environment. In simple terms, on premises means the system runs on computers and rather than sending case files into a general public AI service. That gives the firm greater control over where information is stored, who can access it, how long it is kept, and whether it can leave the firm's environment. That matters for client confidentiality, legal professional privilege, and data protection. Some optional functions such as telephony can involve carefully selected external providers. Those are separate data flows and must be assessed and configured properly for each firm. Lylo is being built around controlled handling of legal information, source traceability, and solicitor review, not casual uploading of sensitive documents into public AI tools." },
    { who:'caller', speaker:'Caller', start:87.66, text:'Cool.' },
    { who:'lylo', speaker:'Lylo', start:90.05, text:'Lylo keeps the firm in control of sensitive legal information. Would you like to try the out of hours receptionist demo?' },
    { who:'caller', speaker:'Caller', start:95.26, text:'And actually, I would like to ask, can you tell me a joke?' },
    { who:'lylo', speaker:'Lylo', start:102.69, text:'Why did the solicitor bring a ladder to court? To reach the higher authority.' },
    { who:'caller', speaker:'Caller', start:106.08, text:"Why? That's not really that funny. We do have everyone." },
    { who:'lylo', speaker:'Lylo', start:113.94, text:'Fair point. That joke should be struck from the record. Why did the lawyer wear 2 pairs of trousers in case they lost their case?' },
    { who:'caller', speaker:'Caller', start:125.03, text:'Okay. Good 1.' },
    { who:'lylo', speaker:'Lylo', start:128.03, text:"I'll take that win." },
    { who:'caller', speaker:'Caller', start:131.74, text:'Can you end the call?' },
    { who:'lylo', speaker:'Lylo', start:134.39, text:'Goodbye.' }
  ].map((turn, index, all) => ({
    ...turn,
    end: turn.end ?? (index < all.length - 1 ? Math.max(turn.start + 0.35, all[index + 1].start - 0.12) : FALLBACK_DURATION)
  }));

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
    const nodes = { line, copy };
    lineNodes.set(index, nodes);
    return nodes;
  };

  const scrollTranscript = (node) => {
    if (!node) return;
    const wanted = Math.max(0, node.offsetTop + node.offsetHeight - transcript.clientHeight + 14);
    if (Math.abs(transcript.scrollTop - wanted) > 1) transcript.scrollTop = wanted;
  };

  const renderTranscript = (time) => {
    let activeIndex = -1;

    turns.forEach((turn, index) => {
      if (time < turn.start) return;
      const { line, copy } = ensureLine(index);

      const availableDuration = Math.max(0.25, turn.end - turn.start);
      const naturalTypingDuration = Math.max(0.35, turn.text.length / TYPE_CHARS_PER_SECOND);
      const typingDuration = Math.min(availableDuration, naturalTypingDuration);
      const progress = Math.max(0, Math.min(1, (time - turn.start) / typingDuration));
      const chars = Math.max(1, Math.floor(turn.text.length * progress));
      copy.textContent = turn.text.slice(0, chars);

      const active = time >= turn.start && time < turn.end;
      line.classList.toggle('active', active);
      if (active) activeIndex = index;
    });

    if (activeIndex < 0) {
      for (let i = turns.length - 1; i >= 0; i--) {
        if (time >= turns[i].start) { activeIndex = i; break; }
      }
    }

    scrollTranscript(lineNodes.get(activeIndex)?.line);
  };

  const updateWave = (time) => {
    const bars = Array.from(wave.children);
    if (!bars.length) return;
    const duration = Number.isFinite(mediaDuration) && mediaDuration > 0 ? mediaDuration : FALLBACK_DURATION;
    const progress = Math.max(0, Math.min(1, time / duration));
    bars.forEach((bar, index) => {
      const barPoint = bars.length === 1 ? 0 : index / (bars.length - 1);
      bar.classList.toggle('active', barPoint <= progress);
    });
  };

  const sync = () => {
    const time = audio.currentTime || 0;
    renderTranscript(time);
    updateWave(time);
  };

  const tick = () => {
    sync();
    if (!audio.paused && !audio.ended) frame = requestAnimationFrame(tick);
  };

  const startVisuals = () => {
    demo.classList.add('started','playing');
    stage.classList.add('phone-demo-active');
    setIcon(true);
    cancelAnimationFrame(frame);
    sync();
    frame = requestAnimationFrame(tick);
  };

  const stopVisuals = () => {
    demo.classList.remove('playing');
    setIcon(false);
    cancelAnimationFrame(frame);
    sync();
  };

  const captureDuration = () => {
    if (Number.isFinite(audio.duration) && audio.duration > 0) mediaDuration = audio.duration;
    sync();
  };

  resetTranscript();
  setIcon(false);
  updateWave(0);

  button.addEventListener('click', async () => {
    if (!audio.paused) {
      audio.pause();
      return;
    }

    if (audio.ended || audio.currentTime >= mediaDuration - 0.15) {
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

  audio.addEventListener('loadedmetadata', captureDuration);
  audio.addEventListener('durationchange', captureDuration);
  audio.addEventListener('timeupdate', sync);
  audio.addEventListener('play', startVisuals);
  audio.addEventListener('pause', () => { if (!audio.ended) stopVisuals(); });
  audio.addEventListener('seeking', sync);
  audio.addEventListener('seeked', sync);
  audio.addEventListener('ended', () => {
    stopVisuals();
    renderTranscript(mediaDuration);
    updateWave(mediaDuration);
  });
})();