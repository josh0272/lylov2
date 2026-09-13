(() => {
  const mq = window.matchMedia('(max-width: 979px)');
  const isMobile = () => mq.matches;

  const safePlay = (video) => {
    video.muted = true;
    video.defaultMuted = true;
    video.loop = true;
    video.playsInline = true;
    video.setAttribute('playsinline', '');

    const attempt = video.play();
    if (attempt && typeof attempt.catch === 'function') {
      attempt.catch(() => {
        if (video.readyState < 2) {
          video.addEventListener('canplay', () => safePlay(video), { once: true });
        }
      });
    }
  };

  const initDesktopVideos = (videos) => {
    videos.forEach((video) => {
      video.autoplay = false;
      video.removeAttribute('autoplay');
      video.muted = true;
      video.defaultMuted = true;
      video.loop = true;
      video.controls = true;
      video.pause();
    });

    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        const video = entry.target;

        if (entry.isIntersecting && entry.intersectionRatio >= 0.35 && !document.hidden) {
          safePlay(video);
        } else if (!video.paused) {
          video.pause();
        }
      });
    }, {
      threshold: [0, 0.35, 0.6, 1]
    });

    videos.forEach((video) => observer.observe(video));

    document.addEventListener('visibilitychange', () => {
      if (document.hidden) {
        videos.forEach((video) => video.pause());
      } else {
        videos.forEach((video) => {
          const rect = video.getBoundingClientRect();
          const vh = window.innerHeight || document.documentElement.clientHeight;
          const visiblePx = Math.max(0, Math.min(rect.bottom, vh) - Math.max(rect.top, 0));
          const ratio = rect.height ? visiblePx / rect.height : 0;
          if (ratio >= 0.35) safePlay(video);
        });
      }
    });
  };

  const initMobileVideos = (videos) => {
    videos.forEach((video) => {
      video.autoplay = false;
      video.removeAttribute('autoplay');
      video.muted = true;
      video.defaultMuted = true;
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

        if (video.dataset.userControlled === '1' && video.paused) return;
        if (video.paused) safePlay(video);
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
  };

  const initMobileCarousel = () => {
    const carousel = document.querySelector('.reg-cards');
    if (!carousel || carousel.dataset.mobileTimerInit) return;

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
  };

  const init = () => {
    document.querySelectorAll('.privacy-node').forEach((node) => {
      node.style.setProperty('animation', 'none', 'important');
    });

    const videos = Array.from(document.querySelectorAll('.demo-media video'));

    if (isMobile()) {
      initMobileVideos(videos);
      initMobileCarousel();
    } else {
      initDesktopVideos(videos);
    }
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init, { once: true });
  } else {
    init();
  }
})();