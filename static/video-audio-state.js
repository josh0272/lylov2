(() => {
  const init = () => {
    const videos = Array.from(document.querySelectorAll('.demo-media video'));
    if (!videos.length) return;

    const markSoundOn = (video) => {
      video.dataset.lyloSoundChosen = '1';
      video.muted = false;
      video.defaultMuted = false;
      video.volume = Math.max(video.volume || 0, 1);
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
      if (video.volume === 0) video.volume = 1;
    };

    const fullscreenVideo = () => document.fullscreenElement || document.webkitFullscreenElement;

    const handleFullscreenChange = () => {
      const active = fullscreenVideo();
      videos.forEach((video) => {
        if (active === video) {
          markSoundOn(video);
        } else if (!active) {
          restoreChosenSound(video);
        }
      });
    };

    document.addEventListener('fullscreenchange', handleFullscreenChange);
    document.addEventListener('webkitfullscreenchange', handleFullscreenChange);

    videos.forEach((video) => {
      let internalReset = false;

      video.addEventListener('webkitbeginfullscreen', () => markSoundOn(video));
      video.addEventListener('webkitendfullscreen', () => {
        window.setTimeout(() => restoreChosenSound(video), 0);
      });

      video.addEventListener('volumechange', () => {
        if (internalReset) return;
        if (!video.muted && video.volume > 0) {
          video.dataset.lyloSoundChosen = '1';
        } else if (video.muted && !fullscreenVideo()) {
          delete video.dataset.lyloSoundChosen;
        }
      });

      video.addEventListener('play', () => {
        window.setTimeout(() => restoreChosenSound(video), 0);
      });

      const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
          if (entry.target !== video) return;
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