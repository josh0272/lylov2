(() => {
  const init = () => {
    const videos = Array.from(document.querySelectorAll('.demo-media video'));
    if (!videos.length) return;

    const ICON_SOUND_ON = `
      <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
        <path d="M4 9v6h4l5 4V5L8 9H4z" fill="currentColor"/>
        <path d="M16 8.5c1.2 1 1.8 2.2 1.8 3.5s-.6 2.5-1.8 3.5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
        <path d="M18.8 6.2c1.8 1.6 2.7 3.5 2.7 5.8s-.9 4.2-2.7 5.8" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
      </svg>`;

    const ICON_MUTED = `
      <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
        <path d="M4 9v6h4l5 4V5L8 9H4z" fill="currentColor"/>
        <path d="M16 9l5 5M21 9l-5 5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
      </svg>`;

    if (!document.getElementById('lylo-video-sound-styles')) {
      const style = document.createElement('style');
      style.id = 'lylo-video-sound-styles';
      style.textContent = `
        .demo-media .lylo-sound-toggle {
          display: none;
        }
        @media (max-width: 979px) {
          .demo-media .lylo-sound-toggle,
          .demo-media .lylo-cc-toggle {
            top: 9px !important;
            height: 32px !important;
            min-width: 32px !important;
            border-radius: 8px !important;
            border: 1px solid rgba(255,255,255,.10) !important;
            background: rgba(4,9,16,.30) !important;
            color: rgba(245,247,250,.88) !important;
            box-shadow: none !important;
            opacity: .76;
            -webkit-backdrop-filter: blur(6px) !important;
            backdrop-filter: blur(6px) !important;
            transition: opacity .18s ease, background .18s ease, border-color .18s ease !important;
          }
          .demo-media .lylo-sound-toggle {
            position: absolute;
            left: calc(50% - 39px);
            right: auto;
            z-index: 9;
            width: 32px;
            padding: 0;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
          }
          .demo-media .lylo-sound-toggle svg {
            width: 17px;
            height: 17px;
            display: block;
          }
          .demo-media .lylo-cc-toggle {
            left: calc(50% + 5px) !important;
            right: auto !important;
            width: 36px !important;
            padding: 0 !important;
            transform: none !important;
          }
          .demo-media .lylo-sound-toggle:active,
          .demo-media .lylo-cc-toggle:active,
          .demo-media .lylo-sound-toggle[aria-pressed="true"],
          .demo-media .lylo-cc-toggle[aria-pressed="true"] {
            background: rgba(4,9,16,.48) !important;
            color: rgba(255,255,255,.98) !important;
            border-color: rgba(255,255,255,.18) !important;
            opacity: .94;
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
      button.innerHTML = soundOn ? ICON_SOUND_ON : ICON_MUTED;
      button.setAttribute('aria-pressed', soundOn ? 'true' : 'false');
      button.setAttribute('aria-label', soundOn ? 'Mute video' : 'Unmute video');
      button.title = soundOn ? 'Mute' : 'Unmute';
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
        button.setAttribute('aria-label', 'Unmute video');
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