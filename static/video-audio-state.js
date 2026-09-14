(() => {
  const init = () => {
    const videos = Array.from(document.querySelectorAll('.demo-media video'));
    if (!videos.length) return;

    const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent) ||
      (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);

    if (isIOS) document.documentElement.classList.add('lylo-ios');

    if (!document.getElementById('lylo-mobile-control-polish')) {
      const style = document.createElement('style');
      style.id = 'lylo-mobile-control-polish';
      style.textContent = `
        .demo-media .lylo-sound-toggle {
          display: none !important;
        }

        .demo-media .lylo-cc-toggle {
          border: 1px solid rgba(255,255,255,.12) !important;
          background: rgba(4,9,16,.30) !important;
          color: rgba(255,255,255,.90) !important;
          box-shadow: none !important;
          opacity: .78 !important;
          -webkit-backdrop-filter: blur(6px) !important;
          backdrop-filter: blur(6px) !important;
          transition: opacity .18s ease, background .18s ease, border-color .18s ease !important;
        }

        .demo-media .lylo-cc-toggle:hover {
          transform: none !important;
          opacity: .96 !important;
          background: rgba(4,9,16,.42) !important;
          border-color: rgba(255,255,255,.20) !important;
        }

        .demo-media .lylo-cc-toggle[aria-pressed="true"] {
          background: rgba(4,9,16,.40) !important;
          color: rgba(255,255,255,.98) !important;
          border-color: rgba(255,255,255,.18) !important;
          opacity: .92 !important;
        }

        .demo-media .lylo-cc-toggle:active {
          opacity: 1 !important;
          background: rgba(4,9,16,.52) !important;
        }

        @media (max-width: 979px) {
          .lylo-ios .demo-media .lylo-cc-toggle {
            display: none !important;
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

    if (!document.getElementById('lylo-phone-demo-script')) {
      const script = document.createElement('script');
      script.id = 'lylo-phone-demo-script';
      script.src = '/static/phone-demo.js?v=1';
      script.defer = true;
      document.body.appendChild(script);
    }
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init, { once: true });
  } else {
    init();
  }
})();