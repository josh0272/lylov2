(() => {
  const mq = window.matchMedia('(max-width: 979px)');
  const isMobile = () => mq.matches;

  const isVideoFullscreen = (video) => {
    const fullscreenElement = document.fullscreenElement || document.webkitFullscreenElement;
    return fullscreenElement === video || video.webkitDisplayingFullscreen === true || video.dataset.lyloFullscreenActive === '1';
  };

  const unmuteFullscreenVideo = (video) => {
    video.dataset.lyloFullscreenAudio = '1';
    video.dataset.lyloSoundChosen = '1';
    video.dataset.lyloFullscreenActive = '1';
    video.muted = false;
    video.defaultMuted = false;
    video.volume = 1;
  };

  const initFullscreenAudio = (videos) => {
    const handleDocumentFullscreen = () => {
      const fullscreenElement = document.fullscreenElement || document.webkitFullscreenElement;

      videos.forEach((video) => {
        if (fullscreenElement === video) {
          unmuteFullscreenVideo(video);
        } else if (!fullscreenElement && video.dataset.lyloFullscreenActive === '1' && video.webkitDisplayingFullscreen !== true) {
          delete video.dataset.lyloFullscreenActive;
        }
      });
    };

    document.addEventListener('fullscreenchange', handleDocumentFullscreen);
    document.addEventListener('webkitfullscreenchange', handleDocumentFullscreen);

    videos.forEach((video) => {
      video.addEventListener('webkitbeginfullscreen', () => unmuteFullscreenVideo(video));
      video.addEventListener('webkitendfullscreen', () => {
        delete video.dataset.lyloFullscreenActive;
      });
    });
  };

  const initCaptionControls = (videos) => {
    if (!document.getElementById('lylo-caption-styles')) {
      const style = document.createElement('style');
      style.id = 'lylo-caption-styles';
      style.textContent = `
        .demo-media .lylo-cc-toggle {
          position: absolute;
          top: 12px;
          right: 12px;
          z-index: 8;
          min-width: 42px;
          height: 32px;
          padding: 0 10px;
          display: inline-flex;
          align-items: center;
          justify-content: center;
          border-radius: 9px;
          border: 1px solid rgba(255,255,255,.18);
          background: rgba(4,9,16,.68);
          color: rgba(245,247,250,.88);
          font: 700 11px/1 -apple-system,BlinkMacSystemFont,"SF Pro Text","Helvetica Neue","Segoe UI",sans-serif;
          letter-spacing: .08em;
          cursor: pointer;
          backdrop-filter: blur(10px);
          -webkit-backdrop-filter: blur(10px);
          box-shadow: 0 7px 22px rgba(0,0,0,.24);
          transition: background .2s ease,color .2s ease,border-color .2s ease,transform .2s ease;
        }
        .demo-media .lylo-cc-toggle:hover {
          transform: translateY(-1px);
          border-color: rgba(255,255,255,.28);
        }
        .demo-media .lylo-cc-toggle[aria-pressed="true"] {
          background: rgba(245,247,250,.94);
          color: #08101d;
          border-color: rgba(255,255,255,.78);
        }
        .demo-media video::cue {
          color: #fff;
          background: rgba(2,6,11,.80);
          font-family: -apple-system,BlinkMacSystemFont,"SF Pro Text","Helvetica Neue","Segoe UI",sans-serif;
        }
        @media (max-width: 979px) {
          .demo-media .lylo-cc-toggle {
            top: 9px;
            right: 9px;
            min-width: 39px;
            height: 30px;
            border-radius: 8px;
          }
        }
      `;
      document.head.appendChild(style);
    }

    const configs = [
      { match: '/static/poc.mp4', src: '/static/poc-en.vtt', id: 'poc' },
      { match: '/static/schedule-of-loss.mp4', src: '/static/schedule-of-loss-en.vtt', id: 'schedule' }
    ];

    videos.forEach((video) => {
      const sourcePaths = Array.from(video.querySelectorAll('source')).map((source) => source.getAttribute('src') || '');
      const config = configs.find((item) => sourcePaths.some((src) => src.includes(item.match)));
      if (!config || video.dataset.lyloCaptionsInit === '1') return;

      video.dataset.lyloCaptionsInit = '1';

      const track = document.createElement('track');
      track.kind = 'captions';
      track.label = 'English';
      track.srclang = 'en';
      track.src = config.src;
      track.default = true;
      track.dataset.lyloCaptionTrack = config.id;
      video.appendChild(track);

      const media = video.closest('.demo-media');
      if (!media) return;

      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'lylo-cc-toggle';
      button.textContent = 'CC';
      button.setAttribute('aria-label', 'Hide captions');
      button.setAttribute('aria-pressed', 'true');
      media.appendChild(button);

      let captionsOn = true;

      const getTextTrack = () => track.track || Array.from(video.textTracks || []).find((item) => item.language === 'en');

      const applyCaptionState = () => {
        const textTrack = getTextTrack();
        if (textTrack) textTrack.mode = captionsOn ? 'showing' : 'hidden';
        button.setAttribute('aria-pressed', captionsOn ? 'true' : 'false');
        button.setAttribute('aria-label', captionsOn ? 'Hide captions' : 'Show captions');
        button.title = captionsOn ? 'Captions on' : 'Captions off';
      };

      button.addEventListener('click', (event) => {
        event.preventDefault();
        event.stopPropagation();
        captionsOn = !captionsOn;
        applyCaptionState();
      });

      track.addEventListener('load', applyCaptionState);
      video.addEventListener('loadedmetadata', applyCaptionState, { once: true });
      window.setTimeout(applyCaptionState, 250);
    });
  };

  const safePlay = (video) => {
    if (!isVideoFullscreen(video) && video.dataset.lyloSoundChosen !== '1') {
      video.muted = true;
      video.defaultMuted = true;
    }
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
        } else if (!video.paused && !isVideoFullscreen(video)) {
          video.pause();
        }
      });
    }, {
      threshold: [0, 0.35, 0.6, 1]
    });

    videos.forEach((video) => observer.observe(video));

    document.addEventListener('visibilitychange', () => {
      if (document.hidden) {
        videos.forEach((video) => {
          if (!isVideoFullscreen(video)) video.pause();
        });
      } else {
        videos.forEach((video) => {
          if (isVideoFullscreen(video)) return;
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
      video.controls = true;
      video.pause();
      video.dataset.userControlled = '1';
    });

    let videoTicking = false;

    const updateActiveVideo = () => {
      videoTicking = false;

      const fullscreenVideos = videos.filter((video) => isVideoFullscreen(video));
      if (fullscreenVideos.length) {
        fullscreenVideos.forEach((video) => {
          if (video.paused) safePlay(video);
        });
        return;
      }

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
    window.addEventListener('resize', () => {
      if (videos.some((video) => isVideoFullscreen(video))) return;
      queueVideoUpdate();
    }, { passive: true });
    window.addEventListener('orientationchange', () => {
      if (videos.some((video) => isVideoFullscreen(video))) return;
      window.setTimeout(queueVideoUpdate, 150);
    }, { passive: true });
    document.addEventListener('visibilitychange', () => {
      if (videos.some((video) => isVideoFullscreen(video))) return;
      queueVideoUpdate();
    });
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

  const loadChat = () => {
    if (document.getElementById('lylo-chat-script')) return;
    const script = document.createElement('script');
    script.id = 'lylo-chat-script';
    script.src = '/static/lylo-chat.js?v=1';
    script.defer = true;
    document.body.appendChild(script);
  };

  const init = () => {
    document.querySelectorAll('.privacy-node').forEach((node) => {
      node.style.setProperty('animation', 'none', 'important');
    });

    const videos = Array.from(document.querySelectorAll('.demo-media video'));
    initCaptionControls(videos);
    initFullscreenAudio(videos);

    if (isMobile()) {
      initMobileVideos(videos);
      initMobileCarousel();
    } else {
      initDesktopVideos(videos);
    }

    loadChat();
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init, { once: true });
  } else {
    init();
  }
})();