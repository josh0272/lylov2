(() => {
  const style = document.createElement('style');
  style.id = 'lylo-chat-styles';
  style.textContent = `
    #lylo-chat-launcher{position:fixed;right:22px;bottom:22px;z-index:1400;display:inline-flex;align-items:center;justify-content:center;min-height:48px;padding:0 18px;border:1px solid rgba(255,255,255,.14);border-radius:999px;background:#050505;color:#f5f5f5;font:500 14px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;box-shadow:0 18px 50px rgba(0,0,0,.48);cursor:pointer;transition:border-color .2s ease,background .2s ease,transform .2s ease}
    #lylo-chat-launcher:hover{border-color:rgba(255,255,255,.25);background:#0a0a0a;transform:translateY(-1px)}
    #lylo-chat-panel{position:fixed;inset:0;z-index:2000;width:100vw;height:100dvh;display:none;grid-template-rows:auto 1fr auto;overflow:hidden;background:#000;color:#f5f5f5;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
    #lylo-chat-panel.open{display:grid}
    .lylo-chat-head{display:flex;align-items:center;justify-content:space-between;padding:20px 24px;border-bottom:1px solid rgba(255,255,255,.08);background:#000}
    .lylo-chat-brand{display:flex;align-items:center;min-width:0}
    .lylo-chat-title{font-family:"Cormorant Garamond",Georgia,serif;font-size:28px;font-weight:600;line-height:1;color:#fff;letter-spacing:-.02em}
    .lylo-chat-actions{display:flex;align-items:center;gap:8px}.lylo-chat-icon-btn{border:0;background:transparent;color:#777;font-size:12px;padding:8px 10px;border-radius:8px;cursor:pointer;transition:color .2s ease,background .2s ease}.lylo-chat-icon-btn:hover{background:rgba(255,255,255,.055);color:#ddd}
    #lylo-chat-close{font-size:22px;line-height:1;color:#aaa;padding:6px 8px}
    .lylo-chat-messages{overflow-y:auto;padding:34px max(20px,calc((100vw - 760px)/2)) 28px;background:#000;scrollbar-width:thin;scrollbar-color:rgba(255,255,255,.10) transparent}
    .lylo-chat-row{display:flex;margin:0 0 14px}.lylo-chat-row.user{justify-content:flex-end}.lylo-chat-row.assistant{justify-content:flex-start}
    .lylo-chat-bubble{max-width:min(78%,680px);padding:12px 14px;border-radius:16px;font-size:14px;line-height:1.55;white-space:pre-wrap;overflow-wrap:anywhere}
    .lylo-chat-row.assistant .lylo-chat-bubble{color:#e7e7e7;background:#111;border:1px solid rgba(255,255,255,.06);border-bottom-left-radius:5px}
    .lylo-chat-row.user .lylo-chat-bubble{color:#fff;background:#232323;border:1px solid rgba(255,255,255,.08);border-bottom-right-radius:5px}
    .lylo-chat-suggestions{display:flex;flex-wrap:wrap;gap:8px;margin:7px 0 18px}.lylo-chat-suggestion{border:1px solid rgba(255,255,255,.10);border-radius:999px;padding:8px 11px;background:#090909;color:#999;font-size:12px;cursor:pointer;transition:color .2s ease,border-color .2s ease,background .2s ease}.lylo-chat-suggestion:hover{border-color:rgba(255,255,255,.22);background:#111;color:#e6e6e6}
    .lylo-chat-typing{display:inline-flex;gap:4px;align-items:center;min-height:18px}.lylo-chat-typing i{width:5px;height:5px;border-radius:50%;background:#8b8b8b;opacity:.45;animation:lyloTyping 1s infinite ease-in-out}.lylo-chat-typing i:nth-child(2){animation-delay:.14s}.lylo-chat-typing i:nth-child(3){animation-delay:.28s}@keyframes lyloTyping{0%,70%,100%{transform:translateY(0);opacity:.35}35%{transform:translateY(-3px);opacity:.9}}
    .lylo-chat-compose{padding:14px max(18px,calc((100vw - 760px)/2)) 18px;border-top:1px solid rgba(255,255,255,.08);background:#000}
    .lylo-chat-form{display:flex;align-items:flex-end;gap:8px;padding:9px 9px 9px 14px;border:1px solid rgba(255,255,255,.11);border-radius:17px;background:#0b0b0b;transition:border-color .2s ease}.lylo-chat-form:focus-within{border-color:rgba(255,255,255,.22)}
    #lylo-chat-input{flex:1;min-width:0;max-height:140px;resize:none;border:0;outline:0;background:transparent;color:#f5f5f5;font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;padding:6px 0}.lylo-chat-form textarea::placeholder{color:#5f5f5f}
    #lylo-chat-send{flex:0 0 auto;width:36px;height:36px;border:0;border-radius:50%;background:#f3f3f3;color:#050505;font-size:18px;cursor:pointer;display:grid;place-items:center;transition:background .2s ease,transform .2s ease}#lylo-chat-send:hover{background:#fff;transform:translateY(-1px)}#lylo-chat-send:disabled{opacity:.38;cursor:default;transform:none}
    .lylo-chat-note{padding:8px 2px 0;text-align:center;color:#4f4f4f;font-size:10px;line-height:1.35}
    body.lylo-chat-open{overflow:hidden!important}
    @media(max-width:600px){#lylo-chat-launcher{right:14px;bottom:14px;min-height:46px;padding:0 16px}.lylo-chat-head{padding:16px 14px}.lylo-chat-title{font-size:26px}.lylo-chat-messages{padding:22px 12px 18px}.lylo-chat-bubble{max-width:88%;font-size:13px}.lylo-chat-compose{padding:10px 10px calc(12px + env(safe-area-inset-bottom))}.lylo-chat-suggestion{font-size:11px}}
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
        <button type="button" class="lylo-chat-icon-btn" id="lylo-chat-close" aria-label="Close chat">✕</button>
      </div>
    </div>
    <div class="lylo-chat-messages" id="lylo-chat-messages"></div>
    <div class="lylo-chat-compose">
      <form class="lylo-chat-form" id="lylo-chat-form">
        <textarea id="lylo-chat-input" rows="1" maxlength="2500" placeholder="Ask Lylo a question…" aria-label="Message Lylo"></textarea>
        <button id="lylo-chat-send" type="submit" aria-label="Send message">↑</button>
      </form>
      <div class="lylo-chat-note">AI responses are general information and not legal advice.</div>
    </div>
  `;

  document.body.appendChild(launcher);
  document.body.appendChild(panel);

  const messagesEl = panel.querySelector('#lylo-chat-messages');
  const form = panel.querySelector('#lylo-chat-form');
  const input = panel.querySelector('#lylo-chat-input');
  const sendBtn = panel.querySelector('#lylo-chat-send');
  const closeBtn = panel.querySelector('#lylo-chat-close');
  const newBtn = panel.querySelector('#lylo-chat-new');
  let busy = false;
  let previousChatId = sessionStorage.getItem('lyloPreviousChatId') || '';
  let pageScrollY = 0;

  const scrollBottom = () => { messagesEl.scrollTop = messagesEl.scrollHeight; };

  const addBubble = (role, text) => {
    const row = document.createElement('div');
    row.className = `lylo-chat-row ${role}`;
    const bubble = document.createElement('div');
    bubble.className = 'lylo-chat-bubble';
    bubble.textContent = text;
    row.appendChild(bubble);
    messagesEl.appendChild(row);
    scrollBottom();
    return row;
  };

  const addWelcome = () => {
    addBubble('assistant', 'Hi — I’m Lylo. Ask me about Lylo, how it works, or how it could help your firm.');
    const suggestions = document.createElement('div');
    suggestions.className = 'lylo-chat-suggestions';
    ['What can Lylo do?', 'How is client data handled?', 'Can Lylo help with an ET1?'].forEach((text) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'lylo-chat-suggestion';
      btn.textContent = text;
      btn.addEventListener('click', () => sendMessage(text));
      suggestions.appendChild(btn);
    });
    messagesEl.appendChild(suggestions);
  };

  const addTyping = () => {
    const row = document.createElement('div');
    row.className = 'lylo-chat-row assistant';
    row.id = 'lylo-chat-typing-row';
    row.innerHTML = '<div class="lylo-chat-bubble"><span class="lylo-chat-typing"><i></i><i></i><i></i></span></div>';
    messagesEl.appendChild(row);
    scrollBottom();
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

    messagesEl.querySelector('.lylo-chat-suggestions')?.remove();
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

      if (!response.ok || !data.ok) {
        addBubble('assistant', data.error || 'I’m having trouble connecting right now. Please try again in a moment.');
      } else {
        if (data.chatId) {
          previousChatId = data.chatId;
          sessionStorage.setItem('lyloPreviousChatId', previousChatId);
        }
        addBubble('assistant', data.reply || 'I’m sorry, I could not generate a response.');
      }
    } catch (err) {
      typing.remove();
      addBubble('assistant', 'I’m having trouble connecting right now. Please try again in a moment.');
    } finally {
      setBusy(false);
      input.focus();
    }
  };

  const openPanel = () => {
    pageScrollY = window.scrollY || window.pageYOffset || 0;
    panel.classList.add('open');
    document.body.classList.add('lylo-chat-open');
    launcher.setAttribute('aria-expanded', 'true');
    launcher.style.display = 'none';
    window.setTimeout(() => input.focus(), 30);
  };

  const closePanel = () => {
    panel.classList.remove('open');
    document.body.classList.remove('lylo-chat-open');
    launcher.setAttribute('aria-expanded', 'false');
    launcher.style.display = '';
    window.scrollTo(0, pageScrollY);
  };

  launcher.addEventListener('click', openPanel);
  closeBtn.addEventListener('click', closePanel);
  newBtn.addEventListener('click', () => {
    previousChatId = '';
    sessionStorage.removeItem('lyloPreviousChatId');
    messagesEl.innerHTML = '';
    addWelcome();
    input.focus();
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
    input.style.height = Math.min(input.scrollHeight, 140) + 'px';
  });

  addWelcome();
})();