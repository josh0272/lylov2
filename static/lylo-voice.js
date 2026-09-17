(() => {
  const init = () => {
    const button = document.getElementById('lylo-voice-call');
    const status = document.getElementById('lylo-voice-status');
    if (!button) return;

    let vapi = null;
    let assistantId = '';
    let state = 'idle';
    let sdkPromise = null;
    let lastStartFailure = '';

    const sleep = (ms) => new Promise((resolve) => window.setTimeout(resolve, ms));

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
            : 'Browser voice call · No phone number needed'
        );
      }
    };

    const errorText = (value) => {
      if (!value) return '';
      if (typeof value === 'string') return value;
      if (value instanceof Error) return value.message || String(value);
      if (typeof value === 'object') {
        if (typeof value.error === 'string') return value.error;
        if (value.error && typeof value.error.message === 'string') return value.error.message;
        if (typeof value.message === 'string') return value.message;
        if (typeof value.reason === 'string') return value.reason;
        try { return JSON.stringify(value); } catch (_) { return String(value); }
      }
      return String(value);
    };

    const loadVapiClass = async () => {
      if (sdkPromise) return sdkPromise;

      sdkPromise = import('https://esm.sh/@vapi-ai/web@2.7.0?bundle&target=es2022')
        .then((module) => {
          let Vapi = module.default;
          if (Vapi && typeof Vapi !== 'function' && typeof Vapi.default === 'function') {
            Vapi = Vapi.default;
          }
          if (typeof Vapi !== 'function' && typeof module.Vapi === 'function') {
            Vapi = module.Vapi;
          }
          if (typeof Vapi !== 'function') {
            throw new Error('Vapi Web SDK loaded, but its client class was not found.');
          }
          return Vapi;
        })
        .catch((error) => {
          sdkPromise = null;
          throw error;
        });

      return sdkPromise;
    };

    const ensureVapi = async () => {
      if (vapi) return vapi;

      const configResponse = await fetch('/api/lylo-voice-config', { cache: 'no-store' });
      const config = await configResponse.json().catch(() => ({}));

      if (!configResponse.ok || !config.ok || !config.publicKey || !config.assistantId) {
        throw new Error(config.error || 'Lylo voice calling is not connected yet.');
      }

      assistantId = config.assistantId;
      const Vapi = await loadVapiClass();
      vapi = new Vapi(config.publicKey);

      vapi.on('call-start', () => {
        lastStartFailure = '';
        setState('active', 'Connected — speak to Lylo');
      });

      vapi.on('call-end', () => {
        if (state !== 'ending') {
          setState('idle', 'Call ended · tap to speak again');
        }
      });

      vapi.on('call-start-progress', (event) => {
        console.log('Lylo call start progress:', event);
      });

      vapi.on('call-start-failed', (event) => {
        const detail = errorText(event && (event.error || event));
        lastStartFailure = detail || 'The browser could not start the voice session.';
        console.error('Lylo call start failed:', event);
        setState('idle', `Call failed: ${lastStartFailure}`);
      });

      vapi.on('error', (error) => {
        const detail = errorText(error);
        console.error('Lylo voice call error:', error);
        if (state === 'connecting') {
          lastStartFailure = detail || lastStartFailure;
          setState('idle', `Call failed: ${lastStartFailure || 'Could not start the voice session.'}`);
        }
      });

      return vapi;
    };

    const endCurrentCall = async () => {
      if (!vapi) {
        setState('idle');
        return;
      }

      setState('ending', 'Ending call…');

      try {
        await vapi.stop();
        await sleep(350);
        setState('idle', 'Call ended · tap to speak again');
      } catch (error) {
        console.error('Could not stop Lylo voice call:', error);
        setState('idle', `Call ended · ${errorText(error) || 'tap to speak again'}`);
      }
    };

    button.addEventListener('click', async () => {
      if (state === 'active') {
        await endCurrentCall();
        return;
      }

      if (state === 'connecting' || state === 'ending') return;

      lastStartFailure = '';
      setState('connecting', 'Connecting to Lylo…');

      try {
        const client = await ensureVapi();
        const call = await client.start(assistantId);

        if (!call && state === 'connecting') {
          setState('idle', `Call failed: ${lastStartFailure || 'Vapi did not create a new call.'}`);
        }
      } catch (error) {
        console.error('Could not start Lylo voice call:', error);
        const detail = errorText(error) || lastStartFailure || 'Could not start the voice session.';
        setState('idle', `Call failed: ${detail}`);
      }
    });

    window.addEventListener('pagehide', () => {
      if (vapi && (state === 'active' || state === 'connecting')) {
        vapi.stop().catch(() => {});
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

(() => {
  const s = document.createElement('script');
  s.src = '/static/site-compliance.js?v=6';
  s.defer = true;
  document.head.appendChild(s);
})();

(() => {
  const s = document.createElement('script');
  s.src = '/static/site-additions.js?v=1';
  s.defer = true;
  document.head.appendChild(s);
})();