(() => {
  const mq = window.matchMedia('(max-width: 979px)');
  const isMobile = () => mq.matches;

  const init = () => {
    if (!isMobile()) return;

    const videos = Array.from(document.querySelectorAll('.demo-media video'));

    // Mobile: remove native autoplay races. One controller decides which demo plays.
    videos.forEach((video) => {
      video.autoplay = false;
      video.removeAttribute('autoplay');
      video.muted = true;
      video.playsInline = true;
      video.setAttribute('playsinline', '');
      video.controls = false;
      video.pause();

      if (!video.dataset.mobileTapInit) {
        video.dataset.mobileTapInit = '1';
        const revealControls = () => {
          video.controls = true;
          video.dataset.userControlled = '1';
          video.removeEventListener('touchstart', revealControls);
          video.removeEventListener('click', revealControls);
        };
        video.addEventListener('touchstart', revealControls, { passive: true });
        video.addEventListener('click', revealControls);
      }
    });

    let videoTicking = false;

    const updateActiveVideo = () => {
      videoTicking = false;
      if (!isMobile() || document.hidden) {
        videos.forEach((video) => video.pause());
        return;
      }

      const vh = window.innerHeight || document.documentElement.clientHeight;
      const viewportCenter = vh / 2;
      let bestVideo = null;
      let bestScore = -Infinity;

      videos.forEach((video) => {
        const rect = video.getBoundingClientRect();
        if (!rect.height) return;

        const visiblePx = Math.max(0, Math.min(rect.bottom, vh) - Math.max(rect.top, 0));
        const visibleRatio = visiblePx / rect.height;
        const center = rect.top + rect.height / 2;
        const centerDistance = Math.abs(center - viewportCenter) / vh;
        const score = visibleRatio - centerDistance * 0.12;

        if (visibleRatio >= 0.38 && score > bestScore) {
          bestScore = score;
          bestVideo = video;
        }
      });

      videos.forEach((video) => {
        if (video !== bestVideo) {
          if (!video.paused) video.pause();
          return;
        }

        // Once a user has taken control, respect a manual pause while it remains active.
        if (video.dataset.userControlled === '1' && video.paused) return;

        if (video.paused) {
          const playAttempt = video.play();
          if (playAttempt && typeof playAttempt.catch === 'function') {
            playAttempt.catch(() => {
              if (video.readyState < 2) {
                video.addEventListener('canplay', updateActiveVideo, { once: true });
              }
            });
          }
        }
      });
    };

    const queueVideoUpdate = () => {
      if (videoTicking) return;
      videoTicking = true;
      requestAnimationFrame(updateActiveVideo);
    };

    window.addEventListener('scroll', queueVideoUpdate, { passive: true });
    window.addEventListener('resize', queueVideoUpdate, { passive: true });
    document.addEventListener('visibilitychange', queueVideoUpdate);
    queueVideoUpdate();

    // Mobile carousel: simple continuous timer, independent from desktop drag code.
    const carousel = document.querySelector('.reg-cards');
    if (carousel && !carousel.dataset.mobileTimerInit) {
      carousel.dataset.mobileTimerInit = '1';
      let paused = false;
      let resumeTimer = null;

      const pause = () => {
        paused = true;
        if (resumeTimer) clearTimeout(resumeTimer);
      };

      const resumeSoon = () => {
        if (resumeTimer) clearTimeout(resumeTimer);
        resumeTimer = setTimeout(() => {
          paused = false;
        }, 900);
      };

      carousel.addEventListener('touchstart', pause, { passive: true });
      carousel.addEventListener('touchmove', pause, { passive: true });
      carousel.addEventListener('touchend', resumeSoon, { passive: true });
      carousel.addEventListener('touchcancel', resumeSoon, { passive: true });

      setInterval(() => {
        if (!isMobile() || paused || document.hidden) return;
        const maxScroll = carousel.scrollWidth - carousel.clientWidth;
        if (maxScroll <= 2) return;

        if (carousel.scrollLeft >= maxScroll - 2) {
          carousel.scrollLeft = 0;
        } else {
          carousel.scrollLeft += 1;
        }
      }, 32);
    }
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init, { once: true });
  } else {
    init();
  }
})();