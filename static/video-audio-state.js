(() => {
  const init = () => {
    const videos = Array.from(document.querySelectorAll('.demo-media video'));
    if (!videos.length) return;

    if (!document.getElementById('lylo-video-sound-styles')) {
      const style = document.createElement('style');
      style.id = 'lylo-video-sound-styles';
      style.textContent = `
        .demo-media .lylo-sound-toggle {
          display: none;
        }
        @media (max-width: 979px) {
          .demo-media .lylo-sound-toggle {
            position: absolute;
            top: 9px;
            left: 9px;
            z-index: 9;
            height: 30px;
            padding: 0 11px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 8px;
            border: 1px solid rgba(255,255,255,.18);
            background: rgba(4,9,16,.72);
            color: rgba(245,247,250,.92);
            font: 650 11px/1 -apple-system,BlinkMacSystemFont,"SF Pro Text","Helvetica Neue","Segoe UI",sans-serif;
            cursor: pointer;
            -webkit-backdrop-filter: blur(10px);
            backdrop-filter: blur(10px);
          }
          .demo-media .lylo-sound-toggle[aria-pressed="true"] {
            background: rgba(245,247,250,.94);
            color: #08101d;
            border-color: rgba(255,255,255,.78);
          }
        }
      `;
      document.head.appendChild(style);
    }

    const isFullscreen = (video) => {
      const active = document.fullscreenElement || document.webkitFullscreenElement;
      return active === video || video.webkitDisplayingFullscreen === true || video.dataset.lyloFullscreenActive === '1';
    };

    const syncSoundButton = (video) => {
      const button = video._lyloSoundButton;
      if (!button) return;
      const soundOn = !video.muted && video.volume > 0;
      button.textContent = soundOn ? 'Sound off' : 'Sound on';
      button.setAttribute('aria-pressed', soundOn ? 'true' : 'false');
      button.setAttribute('aria-label', soundOn ? 'Turn sound off' : 'Turn sound on');
    };

    const markSoundOn = (video) => {
      video.dataset.lyloSoundChosen = '1';
      video.muted = false;
      video.defaultMuted = false;
      try { if (video.volume === 0) video.volume = 1; } catch (_) {}
      syncSoundButton(video);
    };

    const resetSound = (video) => {
      delete video.dataset.lyloSoundChosen;
      video.muted = true;
      video.defaultMuted = true;
      syncSoundButton(video);
    };

    const restoreChosenSound = (video) => {
      if (video.dataset.lyloSoundChosen !== '1') {
        syncSoundButton(video);
        return;
      }
      video.muted = false;
      video.defaultMuted = false;
      try { if (video.volume === 0) video.volume = 1; } catch (_) {}
      syncSoundButton(video);
    };

    const handleFullscreenChange = () => {
      const active = document.fullscreenElement || document.webkitFullscreenElement;
      videos.forEach((video) => {
        if (active === video) {
          video.dataset.lyloFullscreenActive = '1';
          markSoundOn(video);
        } else if (!active && video.dataset.lyloFullscreenActive === '1' && video.webkitDisplayingFullscreen !== true) {
          delete video.dataset.lyloFullscreenActive;
          restoreChosenSound(video);
        }
      });
    };

    document.addEventListener('fullscreenchange', handleFullscreenChange);
    document.addEventListener('webkitfullscreenchange', handleFullscreenChange);

    videos.forEach((video) => {
      let internalReset = false;

      const media = video.closest('.demo-media');
      if (media && !media.querySelector('.lylo-sound-toggle')) {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'lylo-sound-toggle';
        button.textContent = 'Sound on';
        button.setAttribute('aria-label', 'Turn sound on');
        button.setAttribute('aria-pressed', 'false');
        media.appendChild(button);
        video._lyloSoundButton = button;

        button.addEventListener('click', (event) => {
          event.preventDefault();
          event.stopPropagation();

          if (video.muted || video.volume === 0) {
            markSoundOn(video);
            if (video.paused) {
              const attempt = video.play();
              if (attempt && typeof attempt.catch === 'function') attempt.catch(() => {});
            }
          } else {
            internalReset = true;
            resetSound(video);
            internalReset = false;
          }
        });
      }

      video.addEventListener('webkitbeginfullscreen', () => {
        video.dataset.lyloFullscreenActive = '1';
        markSoundOn(video);
      });

      video.addEventListener('webkitendfullscreen', () => {
        delete video.dataset.lyloFullscreenActive;
        window.setTimeout(() => restoreChosenSound(video), 0);
      });

      video.addEventListener('volumechange', () => {
        if (internalReset) {
          syncSoundButton(video);
          return;
        }

        if (!video.muted && video.volume > 0) {
          video.dataset.lyloSoundChosen = '1';
        } else if (video.muted && !isFullscreen(video)) {
          delete video.dataset.lyloSoundChosen;
        }
        syncSoundButton(video);
      });

      video.addEventListener('play', () => {
        window.setTimeout(() => restoreChosenSound(video), 0);
      });

      const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
          if (entry.target !== video) return;

          // iPhone native fullscreen can report the inline element as off-screen,
          // especially during a portrait/landscape rotation. Never reset audio there.
          if (isFullscreen(video)) {
            restoreChosenSound(video);
            return;
          }

          if (!entry.isIntersecting || entry.intersectionRatio <= 0.01) {
            internalReset = true;
            resetSound(video);
            internalReset = false;
          } else {
            restoreChosenSound(video);
          }
        });
      }, { threshold: [0, 0.01, 0.15, 0.35, 1] });

      observer.observe(video);
      syncSoundButton(video);
    });
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init, { once: true });
  } else {
    init();
  }
})();