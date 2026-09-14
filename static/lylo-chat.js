(() => {
  const style = document.createElement('style');
  style.id = 'lylo-chat-styles';
  style.textContent = `
    #lylo-chat-launcher{position:fixed;right:22px;bottom:22px;z-index:1400;display:inline-flex;align-items:center;justify-content:center;min-height:48px;padding:0 18px;border:1px solid rgba(255,255,255,.14);border-radius:999px;background:#050505;color:#f5f5f5;font:500 14px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;box-shadow:0 18px 50px rgba(0,0,0,.48);cursor:pointer;transition:border-color .2s ease,background .2s ease,transform .2s ease}
    #lylo-chat-launcher:hover{border-color:rgba(255,255,255,.25);background:#0a0a0a;transform:translateY(-1px)}

    #lylo-chat-panel{position:fixed;right:24px;bottom:24px;z-index:2000;width:min(760px,calc(100vw - 48px));height:min(760px,calc(100dvh - 48px));display:none;grid-template-rows:72px minmax(0,1fr);overflow:hidden;background:#0b0c0d;color:#f5f5f5;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;border:1px solid rgba(255,255,255,.10);border-radius:24px;box-shadow:0 34px 110px rgba(0,0,0,.62)}
    #lylo-chat-panel.open{display:grid}

    .lylo-chat-head{display:flex;align-items:center;justify-content:space-between;padding:0 28px;border-bottom:1px solid rgba(255,255,255,.09);background:#0b0c0d}
    .lylo-chat-brand{display:flex;align-items:baseline;gap:7px;min-width:0}
    .lylo-chat-title{font-family:'Cormorant Garamond',serif;font-size:28px;font-weight:600;line-height:1;color:#f6f6f6;letter-spacing:-.03em}
    .lylo-chat-actions{display:flex;align-items:center;gap:8px}
    .lylo-chat-icon-btn{border:0;background:transparent;color:#777;font-size:12px;padding:8px 10px;border-radius:9px;cursor:pointer;transition:color .2s ease,background .2s ease}
    .lylo-chat-icon-btn:hover{background:rgba(255,255,255,.055);color:#ddd}
    #lylo-chat-close{width:40px;height:40px;display:grid;place-items:center;padding:0;font-size:22px;line-height:1;color:#b7b7b7;border:1px solid rgba(255,255,255,.08);background:#111315}
    #lylo-chat-close:hover{color:#fff;background:#17191b;border-color:rgba(255,255,255,.13)}

    .lylo-chat-main{position:relative;min-height:0;overflow:hidden;background:#0b0c0d}
    .lylo-chat-empty{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:flex-start;padding:8vh 24px 170px;text-align:center;pointer-events:none;transition:opacity .2s ease,visibility .2s ease}
    .lylo-chat-empty.hidden{opacity:0;visibility:hidden}
    .lylo-chat-empty h2{margin:0;max-width:680px;font-family:'Cormorant Garamond',serif;font-size:clamp(38px,4.2vw,58px);line-height:1.04;font-weight:600;letter-spacing:-.03em;color:#f2f2f2}
    .lylo-chat-empty p{margin:24px 0 0;max-width:580px;font-size:clamp(15px,1.2vw,18px);line-height:1.55;color:#8b8d91}

    .lylo-chat-messages{height:100%;overflow-y:auto;padding:30px 34px 190px;background:#0b0c0d;scrollbar-width:thin;scrollbar-color:rgba(255,255,255,.10) transparent}
    .lylo-chat-row{display:flex;width:100%;margin:0 0 18px}
    .lylo-chat-row.user{justify-content:flex-end}
    .lylo-chat-row.assistant{justify-content:flex-start}
    .lylo-chat-bubble{max-width:min(82%,650px);padding:12px 15px;border-radius:18px;font-size:15px;line-height:1.65;white-space:pre-wrap;overflow-wrap:anywhere}
    .lylo-chat-row.assistant .lylo-chat-bubble{color:#e9e9e9;background:transparent;border:0;padding-left:2px;border-radius:0}
    .lylo-chat-row.user .lylo-chat-bubble{color:#f4f4f4;background:#1a1c1e;border:1px solid rgba(255,255,255,.08);border-bottom-right-radius:6px}

    .lylo-chat-typing{display:inline-flex;gap:5px;align-items:center;min-height:18px}
    .lylo-chat-typing i{width:5px;height:5px;border-radius:50%;background:#8b8b8b;opacity:.45;animation:lyloTyping 1s infinite ease-in-out}
    .lylo-chat-typing i:nth-child(2){animation-delay:.14s}.lylo-chat-typing i:nth-child(3){animation-delay:.28s}
    @keyframes lyloTyping{0%,70%,100%{transform:translateY(0);opacity:.35}35%{transform:translateY(-3px);opacity:.9}}

    .lylo-chat-compose{position:absolute;left:50%;bottom:0;z-index:4;width:calc(100% - 54px);transform:translateX(-50%);padding:0 0 20px;background:linear-gradient(to bottom,rgba(11,12,13,0),#0b0c0d 25%,#0b0c0d 100%)}
    .lylo-chat-form{display:flex;align-items:flex-end;gap:10px;min-height:72px;padding:14px 14px 14px 20px;border:1px solid rgba(255,255,255,.12);border-radius:24px;background:#181a1c;box-shadow:0 16px 50px rgba(0,0,0,.24);transition:border-color .2s ease,background .2s ease}
    .lylo-chat-form:focus-within{border-color:rgba(255,255,255,.20);background:#1b1d1f}
    #lylo-chat-input{flex:1;min-width:0;max-height:150px;resize:none;border:0;outline:0;background:transparent;color:#f5f5f5;font:16px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;padding:10px 0}
    #lylo-chat-input::placeholder{color:#74767a}
    #lylo-chat-send{flex:0 0 auto;width:42px;height:42px;border:0;border-radius:50%;background:#f1f1f1;color:#080808;font-size:19px;cursor:pointer;display:grid;place-items:center;transition:background .2s ease,transform .2s ease}
    #lylo-chat-send:hover{background:#fff;transform:translateY(-1px)}
    #lylo-chat-send:disabled{opacity:.35;cursor:default;transform:none}
    .lylo-chat-note{padding:10px 4px 0;text-align:center;color:#55585c;font-size:10px;line-height:1.35}

    @media(max-width:700px){
      body.lylo-chat-open{overflow:hidden!important}
      #lylo-chat-launcher{right:14px;bottom:14px;min-height:46px;padding:0 16px}
      #lylo-chat-panel{inset:0;width:100vw;height:100dvh;border:0;border-radius:0;box-shadow:none;grid-template-rows:62px minmax(0,1fr)}
      .lylo-chat-head{padding:0 14px}
      .lylo-chat-title{font-family:'Cormorant Garamond',serif;font-size:25px;font-weight:600;letter-spacing:-.03em}
      .lylo-chat-icon-btn{font-size:11px;padding:7px 8px}
      #lylo-chat-close{width:36px;height:36px;font-size:20px}
      .lylo-chat-empty{padding:9vh 20px 155px}
      .lylo-chat-empty h2{font-family:'Cormorant Garamond',serif;font-size:clamp(34px,11vw,52px);font-weight:600;letter-spacing:-.03em}
      .lylo-chat-empty p{margin-top:20px;font-size:15px;max-width:330px}
      .lylo-chat-messages{padding:24px 14px 168px}
      .lylo-chat-bubble{max-width:88%;font-size:14px}
      .lylo-chat-compose{width:calc(100% - 20px);padding-bottom:calc(10px + env(safe-area-inset-bottom))}
      .lylo-chat-form{min-height:60px;border-radius:20px;padding:9px 9px 9px 15px}
      #lylo-chat-input{font-size:16px!important;line-height:1.5;padding:9px 0;-webkit-text-size-adjust:100%}
      #lylo-chat-send{width:38px;height:38px}
      .lylo-chat-note{font-size:9px;padding-top:8px}
    }

    @media(prefers-reduced-motion:reduce){.lylo-chat-typing i{animation:none}}
  `;
  document.head.appendChild(style);

  const launcher = document.createElement('button');
  launcher.id = 'lylo-chat-launcher';
  launcher.type = 'button';
  launcher.setAttribute('aria-expanded', 'false');
  launcher.setAttribute('aria-controls', 'lylo-chat-panel');
  launcher.innerHTML = '<span>Ask Lylo</span>';

  const panel = document.createElement('section');
  panel.id = 'lylo-chat-panel';
  panel.setAttribute('aria-label', 'Chat with Lylo');
  panel.innerHTML = `
    <div class="lylo-chat-head">
      <div class="lylo-chat-brand">
        <div class="lylo-chat-title">Lylo.</div>
      </div>
      <div class="lylo-chat-actions">
        <button type="button" class="lylo-chat-icon-btn" id="lylo-chat-new">New chat</button>
        <button type="button" class="lylo-chat-icon-btn" id="lylo-chat-close" aria-label="Close chat">×</button>
      </div>
    </div>
    <div class="lylo-chat-main">
      <div class="lylo-chat-empty" id="lylo-chat-empty">
        <h2>Ask me about Lylo.</h2>
        <p>How it works, what it can do, or how it could help your firm.</p>
      </div>
      <div class="lylo-chat-messages" id="lylo-chat-messages"></div>
      <div class="lylo-chat-compose">
        <form class="lylo-chat-form" id="lylo-chat-form">
          <textarea id="lylo-chat-input" rows="1" maxlength="2500" placeholder="Ask anything" aria-label="Message Lylo"></textarea>
          <button id="lylo-chat-send" type="submit" aria-label="Send message">↑</button>
        </form>
        <div class="lylo-chat-note">Press Enter to send · AI responses are general information, not legal advice.</div>
      </div>
    </div>
  `;

  document.body.appendChild(launcher);
  document.body.appendChild(panel);

  const messagesEl = panel.querySelector('#lylo-chat-messages');
  const emptyEl = panel.querySelector('#lylo-chat-empty');
  const form = panel.querySelector('#lylo-chat-form');
  const input = panel.querySelector('#lylo-chat-input');
  const sendBtn = panel.querySelector('#lylo-chat-send');
  const closeBtn = panel.querySelector('#lylo-chat-close');
  const newBtn = panel.querySelector('#lylo-chat-new');
  let busy = false;
  let previousChatId = sessionStorage.getItem('lyloPreviousChatId') || '';
  let pageScrollY = 0;

  const scrollBottom = () => { messagesEl.scrollTop = messagesEl.scrollHeight; };
  const updateEmptyState = () => {
    emptyEl.classList.toggle('hidden', messagesEl.children.length > 0);
  };

  const createBubble = (role, text = '') => {
    const row = document.createElement('div');
    row.className = `lylo-chat-row ${role}`;
    const bubble = document.createElement('div');
    bubble.className = 'lylo-chat-bubble';
    bubble.textContent = text;
    row.appendChild(bubble);
    messagesEl.appendChild(row);
    updateEmptyState();
    scrollBottom();
    return {row, bubble};
  };

  const addBubble = (role, text) => createBubble(role, text).row;

  const typeBubble = (text) => new Promise((resolve) => {
    const {bubble} = createBubble('assistant', '');
    const fullText = String(text || '');
    let index = 0;
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    if (reducedMotion || !fullText) {
      bubble.textContent = fullText;
      scrollBottom();
      resolve();
      return;
    }

    const step = () => {
      const remaining = fullText.length - index;
      const chunkSize = remaining > 600 ? 3 : remaining > 250 ? 2 : 1;
      index = Math.min(fullText.length, index + chunkSize);
      bubble.textContent = fullText.slice(0, index);
      scrollBottom();

      if (index < fullText.length) {
        window.setTimeout(step, 14);
      } else {
        resolve();
      }
    };

    step();
  });

  const addTyping = () => {
    const {row, bubble} = createBubble('assistant', '');
    row.id = 'lylo-chat-typing-row';
    bubble.innerHTML = '<span class="lylo-chat-typing"><i></i><i></i><i></i></span>';
    return row;
  };

  const setBusy = (value) => {
    busy = value;
    sendBtn.disabled = value;
    input.disabled = value;
  };

  const sendMessage = async (rawText) => {
    const text = String(rawText || '').trim();
    if (!text || busy) return;

    addBubble('user', text);
    input.value = '';
    input.style.height = 'auto';
    setBusy(true);
    const typing = addTyping();

    try {
      const response = await fetch('/api/lylo-chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          message: text,
          ...(previousChatId ? { previousChatId } : {})
        })
      });
      const data = await response.json().catch(() => ({}));
      typing.remove();
      updateEmptyState();

      if (!response.ok || !data.ok) {
        addBubble('assistant', data.error || 'I’m having trouble connecting right now. Please try again in a moment.');
      } else {
        if (data.chatId) {
          previousChatId = data.chatId;
          sessionStorage.setItem('lyloPreviousChatId', previousChatId);
        }
        await typeBubble(data.reply || 'I’m sorry, I could not generate a response.');
      }
    } catch (err) {
      typing.remove();
      updateEmptyState();
      addBubble('assistant', 'I’m having trouble connecting right now. Please try again in a moment.');
    } finally {
      setBusy(false);
      if (window.innerWidth > 700) input.focus();
    }
  };

  const openPanel = () => {
    if (window.innerWidth <= 700) {
      pageScrollY = window.scrollY || window.pageYOffset || 0;
    }
    panel.classList.add('open');
    document.body.classList.add('lylo-chat-open');
    launcher.setAttribute('aria-expanded', 'true');
    launcher.style.display = 'none';
    updateEmptyState();
    if (window.innerWidth > 700) window.setTimeout(() => input.focus(), 30);
  };

  const closePanel = () => {
    const restoreMobileScroll = window.innerWidth <= 700;
    panel.classList.remove('open');
    document.body.classList.remove('lylo-chat-open');
    launcher.setAttribute('aria-expanded', 'false');
    launcher.style.display = '';
    if (restoreMobileScroll) window.scrollTo(0, pageScrollY);
  };

  launcher.addEventListener('click', openPanel);
  closeBtn.addEventListener('click', closePanel);
  newBtn.addEventListener('click', () => {
    previousChatId = '';
    sessionStorage.removeItem('lyloPreviousChatId');
    messagesEl.innerHTML = '';
    updateEmptyState();
    if (window.innerWidth > 700) input.focus();
  });

  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && panel.classList.contains('open')) closePanel();
  });

  form.addEventListener('submit', (event) => {
    event.preventDefault();
    sendMessage(input.value);
  });

  input.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      form.requestSubmit();
    }
  });

  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 150) + 'px';
  });

  updateEmptyState();
})();