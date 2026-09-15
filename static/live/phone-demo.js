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
  const FALLBACK_DURATION = 281.952;
  const TYPE_CHARS_PER_SECOND = 38;
  let mediaDuration = FALLBACK_DURATION;
  let frame = 0;
  const lineNodes = new Map();

  const controls = demo.querySelector('.audio-controls');
  const timeline = document.createElement('div');
  timeline.className = 'audio-timeline';
  timeline.innerHTML = `
    <span class="audio-time audio-time-current">0:00</span>
    <input class="audio-seek" type="range" min="0" max="1000" step="1" value="0" aria-label="Seek through sample call">
    <span class="audio-time audio-time-total">4:42</span>
  `;
  controls?.insertAdjacentElement('afterend', timeline);

  const seek = timeline.querySelector('.audio-seek');
  const currentTimeLabel = timeline.querySelector('.audio-time-current');
  const totalTimeLabel = timeline.querySelector('.audio-time-total');

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

      .phone-section .audio-timeline {
        width: min(100%, 330px);
        display: grid;
        grid-template-columns: 34px minmax(0,1fr) 34px;
        align-items: center;
        gap: 10px;
        margin: 14px auto 0;
      }

      .phone-section .audio-time {
        color: #68798e;
        font-size: 10px;
        font-weight: 500;
        font-variant-numeric: tabular-nums;
        letter-spacing: .01em;
        line-height: 1;
        user-select: none;
      }

      .phone-section .audio-time-current { text-align: right; }
      .phone-section .audio-time-total { text-align: left; }

      .phone-section .audio-seek {
        --seek: 0%;
        width: 100%;
        height: 18px;
        margin: 0;
        appearance: none;
        -webkit-appearance: none;
        background: transparent;
        cursor: pointer;
        touch-action: pan-x;
      }

      .phone-section .audio-seek::-webkit-slider-runnable-track {
        height: 2px;
        border-radius: 999px;
        background: linear-gradient(90deg, rgba(203,220,241,.72) 0 var(--seek), rgba(139,160,186,.18) var(--seek) 100%);
      }

      .phone-section .audio-seek::-moz-range-track {
        height: 2px;
        border-radius: 999px;
        background: rgba(139,160,186,.18);
      }

      .phone-section .audio-seek::-moz-range-progress {
        height: 2px;
        border-radius: 999px;
        background: rgba(203,220,241,.72);
      }

      .phone-section .audio-seek::-webkit-slider-thumb {
        width: 9px;
        height: 9px;
        margin-top: -3.5px;
        border: 0;
        border-radius: 50%;
        appearance: none;
        -webkit-appearance: none;
        background: #dce8f6;
        box-shadow: 0 0 0 3px rgba(118,157,207,.08);
        opacity: .82;
        transition: opacity .18s ease, transform .18s ease, box-shadow .18s ease;
      }

      .phone-section .audio-seek::-moz-range-thumb {
        width: 9px;
        height: 9px;
        border: 0;
        border-radius: 50%;
        background: #dce8f6;
        box-shadow: 0 0 0 3px rgba(118,157,207,.08);
        opacity: .82;
      }

      .phone-section .audio-seek:hover::-webkit-slider-thumb,
      .phone-section .audio-seek:focus-visible::-webkit-slider-thumb {
        opacity: 1;
        transform: scale(1.08);
        box-shadow: 0 0 0 4px rgba(118,157,207,.12);
      }

      .phone-section .audio-seek:focus-visible { outline: none; }

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

        .phone-section .audio-timeline {
          width: min(88%, 310px);
          margin-top: 12px;
          gap: 9px;
        }
      }
    `;
    document.head.appendChild(style);
  }

  const turns = [
    { who:'lylo', speaker:'Lylo', start:0.06, end:10.80, text:"Hello. You've reached Lylo. I'm an AI assistant built for law firms. You can ask me about Lylo, or try the out-of-hours receptionist demo by speaking as a client. How can I help?" },
    { who:'caller', speaker:'Caller', start:12.72, end:35.72, text:"My employer is harassing and bullying me, and all of my colleagues are acting as if it doesn't happen. And I want to get out of my current employment, but I'm too scared because they are doing illegal things, and I don't want to be blackmailed into doing something I don't want to do." },
    { who:'lylo', speaker:'Lylo', start:37.36, end:48.00, text:"I'm sorry you're dealing with that. I can take some initial details and arrange for a solicitor to speak with you, but I can't provide legal advice myself. Are you a new client or an existing client?" },
    { who:'caller', speaker:'Caller', start:49.02, end:50.08, text:"I'm a new client." },
    { who:'lylo', speaker:'Lylo', start:51.78, end:53.80, text:'Thank you. What is your full name?' },
    { who:'caller', speaker:'Caller', start:55.46, end:57.22, text:'My name is Safa Ahmed.' },
    { who:'lylo', speaker:'Lylo', start:59.64, end:63.48, text:'Thank you, Safa Ahmed. What is the best callback number for you?' },
    { who:'caller', speaker:'Caller', start:65.42, end:71.18, text:'My number is 2 5 0 2 4 0 8 8 8.' },
    { who:'lylo', speaker:'Lylo', start:73.26, end:75.32, text:'Thank you. What is your email address?' },
    { who:'caller', speaker:'Caller', start:77.78, end:82.64, text:'It is Safa at Gmail dot com.' },
    { who:'lylo', speaker:'Lylo', start:84.38, end:87.12, text:'Thank you. Tell me briefly what has been happening at work.' },
    { who:'caller', speaker:'Caller', start:89.44, end:114.46, text:"Um, they discriminate me because of my religion and my clothing. They also call my food a nuisance to them. And I just feel like I'm really being discriminated and harassed in ways that I shouldn't be, according to current employment law rights, protected characteristics." },
    { who:'lylo', speaker:'Lylo', start:116.38, end:116.66, text:"I'm sorry." },
    { who:'caller', speaker:'Caller', start:116.68, end:124.76, text:"I've had a look online, and I know that this shouldn't be happening to me, but it is." },
    { who:'lylo', speaker:'Lylo', start:126.42, end:142.30, text:"I understand. I'll record this as concerns about discrimination and harassment connected to your religion and clothing, including comments about your food. Is the matter urgent? For example, are you at immediate risk, being pressured to do something, or facing a deadline?" },
    { who:'caller', speaker:'Caller', start:144.68, end:156.96, text:"I am. This is urgent. I want to get out, and my mental health is really, really bad. And my children don't have the same mother that they did have before I joined this company." },
    { who:'lylo', speaker:'Lylo', start:158.88, end:179.46, text:"I'm sorry you're going through this, Safa. Your immediate safety comes first. If you feel at risk of harming yourself or you are in immediate danger, call 999 now or contact Samaritans on 116 123. I'll mark this as urgent and ask the team to prioritise a callback. What time is safest for them to call you?" },
    { who:'caller', speaker:'Caller', start:181.70, end:195.34, text:'The time that is safest to call me is before I start work, which is 6 AM till 8 AM, and after work, which is 6 PM till 8 PM.' },
    { who:'lylo', speaker:'Lylo', start:196.80, end:211.82, text:"Thank you. I've recorded your preferred callback windows as 6 to 8 AM and 6 to 8 PM. Would you like to add anything else the solicitor should know, such as your employer's name, any recent incidents, or whether you have kept messages or other evidence?" },
    { who:'caller', speaker:'Caller', start:213.28, end:231.56, text:"I have messages. I have emails as evidence, and they have pictures and stuff that they put on WhatsApp and Facebook groups about me. My employer's name is Brian Lewis, and I can't remember anything else. Right now, I'm too upset." },
    { who:'lylo', speaker:'Lylo', start:233.56, end:262.90, text:"Thank you, Safa. I've recorded that you have messages, emails, and images or posts in WhatsApp and Facebook groups, and that your employer is Brian Lewis. You do not need to go through anything else now. Your enquiry will be marked urgent, with a safe callback requested between 6 to 8 AM or 6 to 8 PM. If you feel at immediate risk, call 999. If you feel you may harm yourself or cannot stay safe, call Samaritans on 116 123." },
    { who:'caller', speaker:'Caller', start:264.80, end:268.06, text:'Amazing. Thank you so much for being so understanding.' },
    { who:'lylo', speaker:'Lylo', start:269.92, end:275.84, text:"You're welcome, Safa. Take care of yourself. The team will use the safe contact details and times you provided." },
    { who:'caller', speaker:'Caller', start:278.20, end:279.52, text:'Okay. Thank you.' },
    { who:'lylo', speaker:'Lylo', start:281.12, end:281.72, text:'Goodbye.' }
  ];

  const formatTime = (seconds) => {
    const safe = Math.max(0, Number.isFinite(seconds) ? seconds : 0);
    const mins = Math.floor(safe / 60);
    const secs = Math.floor(safe % 60).toString().padStart(2, '0');
    return `${mins}:${secs}`;
  };

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

  const updateTimeline = (time) => {
    const duration = Number.isFinite(mediaDuration) && mediaDuration > 0 ? mediaDuration : FALLBACK_DURATION;
    const progress = Math.max(0, Math.min(1, time / duration));
    seek.value = String(Math.round(progress * 1000));
    seek.style.setProperty('--seek', `${progress * 100}%`);
    currentTimeLabel.textContent = formatTime(time);
    totalTimeLabel.textContent = formatTime(duration);
  };

  const sync = () => {
    const time = audio.currentTime || 0;
    renderTranscript(time);
    updateWave(time);
    updateTimeline(time);
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
    updateTimeline(audio.currentTime || 0);
    sync();
  };

  resetTranscript();
  setIcon(false);
  updateWave(0);
  updateTimeline(0);

  seek.addEventListener('input', () => {
    const duration = Number.isFinite(mediaDuration) && mediaDuration > 0 ? mediaDuration : FALLBACK_DURATION;
    const nextTime = (Number(seek.value) / 1000) * duration;
    demo.classList.add('started');
    stage.classList.add('phone-demo-active');
    audio.currentTime = nextTime;
    resetTranscript();
    renderTranscript(nextTime);
    updateWave(nextTime);
    updateTimeline(nextTime);
  });

  button.addEventListener('click', async () => {
    if (!audio.paused) {
      audio.pause();
      return;
    }

    if (audio.ended || audio.currentTime >= mediaDuration - 0.15) {
      audio.currentTime = 0;
      resetTranscript();
      updateWave(0);
      updateTimeline(0);
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
  audio.addEventListener('seeking', () => {
    resetTranscript();
    sync();
  });
  audio.addEventListener('seeked', sync);
  audio.addEventListener('ended', () => {
    stopVisuals();
    renderTranscript(mediaDuration);
    updateWave(mediaDuration);
    updateTimeline(mediaDuration);
  });
})();