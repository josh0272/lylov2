(() => {
  const init = () => {
    const videos = Array.from(document.querySelectorAll('.demo-media video'));
    if (!videos.length) return;

    if (!document.getElementById('lylo-mobile-control-polish')) {
      const style = document.createElement('style');
      style.id = 'lylo-mobile-control-polish';
      style.textContent = `
        .demo-media .lylo-sound-toggle {
          display: none !important;
        }

        @media (max-width: 979px) {
          .demo-media .lylo-cc-toggle {
            top: 50px !important;
            right: 10px !important;
            left: auto !important;
            width: 36px !important;
            min-width: 36px !important;
            height: 30px !important;
            padding: 0 !important;
            border-radius: 8px !important;
            border: 1px solid rgba(255,255,255,.10) !important;
            background: rgba(4,9,16,.26) !important;
            color: rgba(255,255,255,.90) !important;
            box-shadow: none !important;
            opacity: .72 !important;
            transform: none !important;
            -webkit-backdrop-filter: blur(5px) !important;
            backdrop-filter: blur(5px) !important;
          }

          .demo-media .lylo-cc-toggle[aria-pressed="true"] {
            background: rgba(4,9,16,.34) !important;
            color: rgba(255,255,255,.96) !important;
            border-color: rgba(255,255,255,.14) !important;
            opacity: .84 !important;
          }

          .demo-media .lylo-cc-toggle:active {
            opacity: 1 !important;
            background: rgba(4,9,16,.48) !important;
          }
        }
      `;
      document.head.appendChild(style);
    }

    const isFullscreen = (video) => {
      const active = document.fullscreenElement || document.webkitFullscreenElement;
      return active === video || video.webkitDisplayingFullscreen === true || video.dataset.lyloFullscreenActive === '1';
    };

    const markSoundOn = (video) => {
      video.dataset.lyloSoundChosen = '1';
      video.muted = false;
      video.defaultMuted = false;
      try { if (video.volume === 0) video.volume = 1; } catch (_) {}
    };

    const resetSound = (video) => {
      delete video.dataset.lyloSoundChosen;
      video.muted = true;
      video.defaultMuted = true;
    };

    const restoreChosenSound = (video) => {
      if (video.dataset.lyloSoundChosen !== '1') return;
      video.muted = false;
      video.defaultMuted = false;
      try { if (video.volume === 0) video.volume = 1; } catch (_) {}
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

      video.addEventListener('webkitbeginfullscreen', () => {
        video.dataset.lyloFullscreenActive = '1';
        markSoundOn(video);
      });

      video.addEventListener('webkitendfullscreen', () => {
        delete video.dataset.lyloFullscreenActive;
        window.setTimeout(() => restoreChosenSound(video), 0);
      });

      // Native mobile video controls fire volumechange too, so the iPhone's own
      // mute button becomes the single sound control while we remember its state.
      video.addEventListener('volumechange', () => {
        if (internalReset) return;

        if (!video.muted && video.volume > 0) {
          video.dataset.lyloSoundChosen = '1';
        } else if (video.muted && !isFullscreen(video)) {
          delete video.dataset.lyloSoundChosen;
        }
      });

      video.addEventListener('play', () => {
        window.setTimeout(() => restoreChosenSound(video), 0);
      });

      const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
          if (entry.target !== video) return;

          // Native iPhone fullscreen can report the inline video as off-screen
          // while the phone rotates. Do not change audio until fullscreen ends.
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
    });
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init, { once: true });
  } else {
    init();
  }
})();