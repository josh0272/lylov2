(() => {
  const load = (src, done) => {
    const s = document.createElement('script');
    s.src = src;
    s.defer = true;
    if (done) s.addEventListener('load', done, { once:true });
    document.head.appendChild(s);
  };

  load('/static/preview-cta-base.js?v=1', () => {
    load('/static/preview-team-v2.js?v=1');
  });
})();