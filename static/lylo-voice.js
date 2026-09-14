(() => {
  const init = () => {
    const button = document.getElementById('lylo-voice-call');
    const status = document.getElementById('lylo-voice-status');
    if (!button) return;

    let vapi = null;
    let assistantId = '';
    let state = 'idle';
    let sdkPromise = null;
    let endFallbackTimer = null;

    const setState = (next, message) => {
      state = next;
      button.classList.toggle('is-active', next === 'active');
      button.disabled = next === 'connecting' || next === 'ending';

      if (next === 'connecting') button.textContent = 'Connecting…';
      else if (next === 'ending') button.textContent = 'Ending…';
      else if (next === 'active') button.textContent = 'End call';
      else button.textContent = 'Call Lylo';

      if (status) {
        status.textContent = message || (
          next === 'active'
            ? 'Connected — speak to Lylo'
            : 'Browser voice call · no phone number needed'
        );
      }
    };

    const loadVapiBrowserSDK = () => {
      if (window.vapiSDK && typeof window.vapiSDK.run === 'function') {
        return Promise.resolve(window.vapiSDK);
      }

      if (sdkPromise) return sdkPromise;

      sdkPromise = new Promise((resolve, reject) => {
        const existing = document.querySelector('script[data-lylo-vapi-sdk="1"]');
        if (existing) {
          existing.addEventListener('load', () => {
            if (window.vapiSDK && typeof window.vapiSDK.run === 'function') resolve(window.vapiSDK);
            else reject(new Error('Vapi browser SDK did not initialise.'));
          }, { once: true });
          existing.addEventListener('error', () => reject(new Error('Could not load the Vapi browser SDK.')), { once: true });
          return;
        }

        const script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/gh/VapiAI/html-script-tag@latest/dist/assets/index.js';
        script.async = true;
        script.defer = true;
        script.dataset.lyloVapiSdk = '1';
        script.onload = () => {
          if (window.vapiSDK && typeof window.vapiSDK.run === 'function') resolve(window.vapiSDK);
          else reject(new Error('Vapi browser SDK did not initialise.'));
        };
        script.onerror = () => reject(new Error('Could not load the Vapi browser SDK.'));
        document.head.appendChild(script);
      });

      return sdkPromise;
    };

    const discardClient = (client) => {
      if (!client) return;
      try {
        if (typeof client.removeAllListeners === 'function') client.removeAllListeners();
      } catch (error) {
        console.warn('Could not clear old Lylo call listeners:', error);
      }
      if (vapi === client) vapi = null;
      if (window.vapiSDK && window.vapiSDK.vapi === client) {
        window.vapiSDK.vapi = null;
      }
      document.getElementById('vapi-support-btn')?.remove();
    };

    const ensureVapi = async () => {
      if (vapi) return vapi;

      const configResponse = await fetch('/api/lylo-voice-config', { cache: 'no-store' });
      const config = await configResponse.json().catch(() => ({}));

      if (!configResponse.ok || !config.ok || !config.publicKey || !config.assistantId) {
        throw new Error(config.error || 'Lylo voice calling is not connected yet.');
      }

      assistantId = config.assistantId;

      const sdk = await loadVapiBrowserSDK();
      const client = sdk.run({
        apiKey: config.publicKey,
        assistant: assistantId,
        config: {
          position: 'bottom-right',
          offset: '-9999px',
          width: '1px',
          height: '1px'
        }
      });

      if (!client) {
        throw new Error('Vapi could not initialise the Lylo call.');
      }

      vapi = client;
      document.getElementById('vapi-support-btn')?.remove();

      client.on('call-start', () => {
        if (vapi !== client) return;
        setState('active', 'Connected — speak to Lylo');
      });

      client.on('call-end', () => {
        if (endFallbackTimer) {
          clearTimeout(endFallbackTimer);
          endFallbackTimer = null;
        }
        if (vapi === client) {
          discardClient(client);
          setState('idle', 'Call ended · tap to speak again');
        }
      });

      client.on('error', (error) => {
        console.error('Lylo voice call error:', error);
        if (vapi === client && state !== 'active') {
          discardClient(client);
          setState('idle', 'Could not start the call · try again');
        }
      });

      return client;
    };

    const endCurrentCall = () => {
      const client = vapi;
      if (!client) {
        setState('idle');
        return;
      }

      setState('ending', 'Ending call…');

      try {
        client.stop();
      } catch (error) {
        console.error('Could not stop Lylo voice call:', error);
        discardClient(client);
        setState('idle', 'Call ended · tap to speak again');
        return;
      }

      if (endFallbackTimer) clearTimeout(endFallbackTimer);
      endFallbackTimer = setTimeout(() => {
        if (vapi === client) {
          discardClient(client);
          setState('idle', 'Call ended · tap to speak again');
        }
        endFallbackTimer = null;
      }, 3000);
    };

    button.addEventListener('click', async () => {
      if (state === 'active') {
        endCurrentCall();
        return;
      }

      if (state === 'connecting' || state === 'ending') return;

      setState('connecting', 'Allow microphone access when your browser asks');

      try {
        const client = await ensureVapi();
        await client.start(assistantId);
      } catch (error) {
        console.error('Could not start Lylo voice call:', error);
        const failedClient = vapi;
        discardClient(failedClient);
        const message = error && error.message
          ? error.message
          : 'Could not start the call · try again';
        setState('idle', message);
      }
    });

    window.addEventListener('pagehide', () => {
      if (vapi && (state === 'active' || state === 'connecting')) {
        try { vapi.stop(); } catch (_) {}
      }
    });

    setState('idle');
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init, { once: true });
  } else {
    init();
  }
})();