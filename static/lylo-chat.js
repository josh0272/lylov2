(() => {
  const style = document.createElement('style');
  style.id = 'lylo-chat-styles';
  style.textContent = `
    #lylo-chat-launcher{position:fixed;right:22px;bottom:22px;z-index:1400;display:inline-flex;align-items:center;gap:9px;min-height:48px;padding:0 17px;border:1px solid rgba(255,255,255,.13);border-radius:999px;background:rgba(10,20,34,.94);color:#f4f7fb;font:600 14px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;box-shadow:0 18px 55px rgba(0,0,0,.38);backdrop-filter:blur(16px);cursor:pointer}
    #lylo-chat-launcher:hover{border-color:rgba(143,191,255,.34);background:rgba(13,26,44,.98)}
    #lylo-chat-launcher .lylo-chat-dot{width:8px;height:8px;border-radius:50%;background:#6ad9b0;box-shadow:0 0 16px rgba(106,217,176,.45)}
    #lylo-chat-panel{position:fixed;right:22px;bottom:82px;z-index:1401;width:min(390px,calc(100vw - 32px));height:min(610px,calc(100dvh - 112px));display:none;grid-template-rows:auto 1fr auto;overflow:hidden;border:1px solid rgba(255,255,255,.10);border-radius:22px;background:rgba(7,14,25,.985);box-shadow:0 28px 90px rgba(0,0,0,.56);backdrop-filter:blur(24px);color:#f5f7fa;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
    #lylo-chat-panel.open{display:grid}
    .lylo-chat-head{display:flex;align-items:center;justify-content:space-between;padding:16px 17px;border-bottom:1px solid rgba(255,255,255,.07);background:rgba(255,255,255,.012)}
    .lylo-chat-brand{display:flex;align-items:center;gap:10px;min-width:0}
    .lylo-chat-mark{width:34px;height:34px;border:1px solid rgba(255,255,255,.10);border-radius:11px;display:grid;place-items:center;background:linear-gradient(145deg,rgba(99,168,255,.12),rgba(106,217,176,.05));font-family:"Cormorant Garamond",Georgia,serif;font-size:22px;font-weight:600}
    .lylo-chat-title{font-size:14px;font-weight:650;line-height:1.15}.lylo-chat-status{margin-top:3px;color:#8796a9;font-size:11px;display:flex;align-items:center;gap:5px}.lylo-chat-status::before{content:"";width:6px;height:6px;border-radius:50%;background:#6ad9b0}
    .lylo-chat-actions{display:flex;align-items:center;gap:5px}.lylo-chat-icon-btn{border:0;background:transparent;color:#8e9bad;font-size:12px;padding:7px 8px;border-radius:8px;cursor:pointer}.lylo-chat-icon-btn:hover{background:rgba(255,255,255,.05);color:#eef4fb}
    .lylo-chat-messages{overflow-y:auto;padding:18px 14px 16px;scrollbar-width:thin;scrollbar-color:rgba(255,255,255,.10) transparent}
    .lylo-chat-row{display:flex;margin:0 0 11px}.lylo-chat-row.user{justify-content:flex-end}.lylo-chat-row.assistant{justify-content:flex-start}
    .lylo-chat-bubble{max-width:84%;padding:10px 12px;border-radius:15px;font-size:13px;line-height:1.5;white-space:pre-wrap;overflow-wrap:anywhere}
    .lylo-chat-row.assistant .lylo-chat-bubble{color:#dce5ef;background:rgba(255,255,255,.045);border:1px solid rgba(255,255,255,.055);border-bottom-left-radius:5px}
    .lylo-chat-row.user .lylo-chat-bubble{color:#f8fbff;background:linear-gradient(145deg,rgba(46,94,150,.68),rgba(31,73,122,.72));border:1px solid rgba(129,184,250,.18);border-bottom-right-radius:5px}
    .lylo-chat-suggestions{display:flex;flex-wrap:wrap;gap:7px;margin:4px 0 14px}.lylo-chat-suggestion{border:1px solid rgba(255,255,255,.08);border-radius:999px;padding:7px 10px;background:rgba(255,255,255,.022);color:#aeb9c8;font-size:11px;cursor:pointer}.lylo-chat-suggestion:hover{border-color:rgba(125,181,247,.22);color:#e8f0f9}
    .lylo-chat-typing{display:inline-flex;gap:4px;align-items:center;min-height:18px}.lylo-chat-typing i{width:5px;height:5px;border-radius:50%;background:#91a1b5;opacity:.45;animation:lyloTyping 1s infinite ease-in-out}.lylo-chat-typing i:nth-child(2){animation-delay:.14s}.lylo-chat-typing i:nth-child(3){animation-delay:.28s}@keyframes lyloTyping{0%,70%,100%{transform:translateY(0);opacity:.35}35%{transform:translateY(-3px);opacity:.9}}
    .lylo-chat-compose{padding:11px;border-top:1px solid rgba(255,255,255,.07);background:rgba(4,9,17,.72)}
    .lylo-chat-form{display:flex;align-items:flex-end;gap:8px;padding:7px 7px 7px 12px;border:1px solid rgba(255,255,255,.09);border-radius:16px;background:rgba(255,255,255,.025)}
    #lylo-chat-input{flex:1;min-width:0;max-height:110px;resize:none;border:0;outline:0;background:transparent;color:#f4f7fb;font:13px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;padding:5px 0}.lylo-chat-form textarea::placeholder{color:#69788b}
    #lylo-chat-send{flex:0 0 auto;width:34px;height:34px;border:1px solid rgba(126,183,248,.18);border-radius:11px;background:linear-gradient(145deg,#1d4d7e,#17385d);color:white;font-size:17px;cursor:pointer;display:grid;place-items:center}#lylo-chat-send:disabled{opacity:.45;cursor:default}
    .lylo-chat-note{padding:7px 2px 0;text-align:center;color:#607084;font-size:9.5px;line-height:1.35}
    @media(max-width:600px){#lylo-chat-launcher{right:14px;bottom:14px;min-height:46px;padding:0 15px}#lylo-chat-panel{left:10px;right:10px;bottom:70px;width:auto;height:min(72dvh,590px);border-radius:19px}.lylo-chat-head{padding:14px 15px}.lylo-chat-messages{padding:15px 12px}.lylo-chat-bubble{max-width:88%}}
    @media(prefers-reduced-motion:reduce){.lylo-chat-typing i{animation:none}}
  `;
  document.head.appendChild(style);

  const launcher = document.createElement('button');
  launcher.id = 'lylo-chat-launcher';
  launcher.type = 'button';
  launcher.setAttribute('aria-expanded', 'false');
  launcher.setAttribute('aria-controls', 'lylo-chat-panel');
  launcher.innerHTML = '<span class="lylo-chat-dot"></span><span>Ask Lylo</span>';

  const panel = document.createElement('section');
  panel.id = 'lylo-chat-panel';
  panel.setAttribute('aria-label', 'Chat with Lylo');
  panel.innerHTML = `
    <div class="lylo-chat-head">
      <div class="lylo-chat-brand">
        <div class="lylo-chat-mark">L</div>
        <div><div class="lylo-chat-title">Lylo.</div><div class="lylo-chat-status">AI assistant</div></div>
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