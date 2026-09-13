(() => {
  const style = document.createElement('style');
  style.id = 'lylo-chat-styles';
  style.textContent = `
    #lylo-chat-launcher{position:fixed;right:22px;bottom:22px;z-index:1400;display:inline-flex;align-items:center;justify-content:center;min-height:48px;padding:0 18px;border:1px solid rgba(255,255,255,.14);border-radius:999px;background:#050505;color:#f5f5f5;font:500 14px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;box-shadow:0 18px 50px rgba(0,0,0,.48);cursor:pointer;transition:border-color .2s ease,background .2s ease,transform .2s ease}
    #lylo-chat-launcher:hover{border-color:rgba(255,255,255,.25);background:#0a0a0a;transform:translateY(-1px)}
    #lylo-chat-panel{position:fixed;right:22px;bottom:82px;z-index:1401;width:min(390px,calc(100vw - 32px));height:min(610px,calc(100dvh - 112px));display:none;grid-template-rows:auto 1fr auto;overflow:hidden;border:1px solid rgba(255,255,255,.10);border-radius:20px;background:#000;box-shadow:0 30px 100px rgba(0,0,0,.68);color:#f5f5f5;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
    #lylo-chat-panel.open{display:grid}
    .lylo-chat-head{display:flex;align-items:center;justify-content:space-between;padding:18px 18px 16px;border-bottom:1px solid rgba(255,255,255,.08);background:#000}
    .lylo-chat-brand{display:flex;align-items:center;min-width:0}
    .lylo-chat-title{font-family:"Cormorant Garamond",Georgia,serif;font-size:24px;font-weight:600;line-height:1;color:#fff;letter-spacing:-.02em}
    .lylo-chat-actions{display:flex;align-items:center;gap:3px}.lylo-chat-icon-btn{border:0;background:transparent;color:#777;font-size:11px;padding:7px 8px;border-radius:8px;cursor:pointer;transition:color .2s ease,background .2s ease}.lylo-chat-icon-btn:hover{background:rgba(255,255,255,.055);color:#ddd}
    .lylo-chat-messages{overflow-y:auto;padding:20px 15px 18px;background:#000;scrollbar-width:thin;scrollbar-color:rgba(255,255,255,.10) transparent}
    .lylo-chat-row{display:flex;margin:0 0 12px}.lylo-chat-row.user{justify-content:flex-end}.lylo-chat-row.assistant{justify-content:flex-start}
    .lylo-chat-bubble{max-width:84%;padding:10px 12px;border-radius:15px;font-size:13px;line-height:1.5;white-space:pre-wrap;overflow-wrap:anywhere}
    .lylo-chat-row.assistant .lylo-chat-bubble{color:#e7e7e7;background:#111;border:1px solid rgba(255,255,255,.06);border-bottom-left-radius:5px}
    .lylo-chat-row.user .lylo-chat-bubble{color:#fff;background:#232323;border:1px solid rgba(255,255,255,.08);border-bottom-right-radius:5px}
    .lylo-chat-suggestions{display:flex;flex-wrap:wrap;gap:7px;margin:5px 0 15px}.lylo-chat-suggestion{border:1px solid rgba(255,255,255,.10);border-radius:999px;padding:7px 10px;background:#090909;color:#999;font-size:11px;cursor:pointer;transition:color .2s ease,border-color .2s ease,background .2s ease}.lylo-chat-suggestion:hover{border-color:rgba(255,255,255,.22);background:#111;color:#e6e6e6}
    .lylo-chat-typing{display:inline-flex;gap:4px;align-items:center;min-height:18px}.lylo-chat-typing i{width:5px;height:5px;border-radius:50%;background:#8b8b8b;opacity:.45;animation:lyloTyping 1s infinite ease-in-out}.lylo-chat-typing i:nth-child(2){animation-delay:.14s}.lylo-chat-typing i:nth-child(3){animation-delay:.28s}@keyframes lyloTyping{0%,70%,100%{transform:translateY(0);opacity:.35}35%{transform:translateY(-3px);opacity:.9}}
    .lylo-chat-compose{padding:11px;border-top:1px solid rgba(255,255,255,.08);background:#000}
    .lylo-chat-form{display:flex;align-items:flex-end;gap:8px;padding:7px 7px 7px 12px;border:1px solid rgba(255,255,255,.11);border-radius:15px;background:#0b0b0b;transition:border-color .2s ease}.lylo-chat-form:focus-within{border-color:rgba(255,255,255,.22)}
    #lylo-chat-input{flex:1;min-width:0;max-height:110px;resize:none;border:0;outline:0;background:transparent;color:#f5f5f5;font:13px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;padding:5px 0}.lylo-chat-form textarea::placeholder{color:#5f5f5f}
    #lylo-chat-send{flex:0 0 auto;width:34px;height:34px;border:0;border-radius:50%;background:#f3f3f3;color:#050505;font-size:17px;cursor:pointer;display:grid;place-items:center;transition:background .2s ease,transform .2s ease}#lylo-chat-send:hover{background:#fff;transform:translateY(-1px)}#lylo-chat-send:disabled{opacity:.38;cursor:default;transform:none}
    .lylo-chat-note{padding:7px 2px 0;text-align:center;color:#4f4f4f;font-size:9.5px;line-height:1.35}
    @media(max-width:600px){#lylo-chat-launcher{right:14px;bottom:14px;min-height:46px;padding:0 16px}#lylo-chat-panel{left:10px;right:10px;bottom:70px;width:auto;height:min(72dvh,590px);border-radius:18px}.lylo-chat-head{padding:16px 15px 14px}.lylo-chat-messages{padding:17px 12px}.lylo-chat-bubble{max-width:88%}}
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
    panel.classList.add('open');
    launcher.setAttribute('aria-expanded', 'true');
    window.setTimeout(() => input.focus(), 30);
  };

  const closePanel = () => {
    panel.classList.remove('open');
    launcher.setAttribute('aria-expanded', 'false');
  };

  launcher.addEventListener('click', () => panel.classList.contains('open') ? closePanel() : openPanel());
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
    input.style.height = Math.min(input.scrollHeight, 110) + 'px';
  });

  addWelcome();
})();