(() => {
  const pilotPath = '/preview/founding-pilot';
  const aboutPath = '/static/preview-about.html';
  const joshPhoto = '/static/preview-joshua.jpg';
  const jessPhoto = '/static/preview-jessica.jpg';

  const setText = (selector, value) => {
    const node = document.querySelector(selector);
    if (node) node.textContent = value;
  };

  const initSectionAnchors = () => {
    const intro = document.querySelector('.demo-intro');
    const warning = document.querySelector('.reg-trust');
    const demoHeading = document.querySelector('.demo-heading');
    if (intro?.id === 'demos') intro.removeAttribute('id');
    if (warning) warning.id = 'sra-warning';
    if (demoHeading) demoHeading.id = 'demos';
  };

  const initDesktopNav = () => {
    const headerInner = document.querySelector('.site-header .header-inner');
    const hamburger = document.getElementById('hamburger');
    if (!headerInner || !hamburger) return;
    let nav = headerInner.querySelector('.desktop-nav');
    if (!nav) {
      nav = document.createElement('nav');
      nav.className = 'desktop-nav';
      nav.setAttribute('aria-label', 'Primary');
      headerInner.insertBefore(nav, hamburger);
    }
    nav.innerHTML = `
      <a href="#sra-warning">SRA warning</a>
      <a href="#demos">Demos</a>
      <a href="#privacy">Private AI</a>
      <a class="desktop-pilot-link" href="${pilotPath}">Founding Pilot</a>`;
  };

  const closeMobileMenu = () => {
    const menu = document.getElementById('mobileMenu');
    const hamburger = document.getElementById('hamburger');
    if (!menu || !hamburger) return;
    menu.classList.remove('open');
    menu.setAttribute('aria-hidden', 'true');
    hamburger.setAttribute('aria-expanded', 'false');
    document.body.style.overflow = '';
  };

  const initMobileNav = () => {
    const panel = document.querySelector('#mobileMenu .panel-inner');
    if (!panel) return;
    panel.innerHTML = `
      <a href="#sra-warning">SRA warning</a>
      <a href="#demos">Demos</a>
      <a href="#privacy">Private AI</a>
      <a href="${pilotPath}">Founding Pilot</a>`;
    panel.querySelectorAll('a').forEach(link => link.addEventListener('click', closeMobileMenu));
  };

  const initCommunitySections = () => {
    const finalSection = document.querySelector('.final');
    if (!finalSection || document.querySelector('.lylo-community')) return;

    const section = document.createElement('section');
    section.className = 'lylo-community';
    section.innerHTML = `
      <div class="lylo-community-inner">
        <article class="lylo-community-card lylo-research-card">
          <div class="lylo-community-kicker">Help shape Lylo</div>
          <h3>Your experience can shape what we build next.</h3>
          <p>If you study or work in law, tell us which tasks take too much time, where AI could help and where it should stay out of the way.</p>
          <div class="lylo-research-meta">Solicitors · trainees · law students</div>
          <a class="lylo-community-button" href="/research">Take the questionnaire <span>→</span></a>
          <div class="lylo-community-note">Short questionnaire · no sales pitch.</div>
        </article>

        <article class="lylo-community-card lylo-team-card">
          <div class="lylo-community-kicker">The people behind Lylo</div>
          <div class="lylo-team-mini">
            <div class="lylo-mini-person">
              <img src="${joshPhoto}" alt="Joshua Sam">
              <div><strong>Joshua Sam</strong><span>Engineering · product & technology</span></div>
            </div>
            <div class="lylo-mini-person">
              <img src="${jessPhoto}" alt="Jessica Jayan">
              <div><strong>Jessica Jayan</strong><span>Scots (Clinical) LLB 2:1 · DPLP · Strathclyde Law Clinic</span></div>
            </div>
          </div>
          <p>One side builds and tests the technology. The other brings legal training and real Law Clinic experience into the workflows we choose to test.</p>
          <a class="lylo-community-button secondary" href="${aboutPath}">Meet the team <span>→</span></a>
        </article>
      </div>`;

    finalSection.parentNode.insertBefore(section, finalSection);
  };

  const initDemoCtas = () => {
    const heroCta = document.querySelector('.hero .cta');
    if (heroCta) {
      heroCta.textContent = 'Book a 20-minute demo';
      heroCta.setAttribute('href', `${pilotPath}#book`);
    }
    setText('.hero .note', 'See Lylo, ask questions, and decide if the pilot is worth testing.');

    const sections = Array.from(document.querySelectorAll('.demo-section'));
    const byHeading = heading => sections.find(section => section.querySelector('.demo-copy h3')?.textContent.trim() === heading);

    const caseCta = byHeading('Ask the case. Get the answer.')?.querySelector('.demo-cta');
    if (caseCta) {
      caseCta.textContent = 'Explore the founding pilot';
      caseCta.setAttribute('href', pilotPath);
    }

    const et1Section = byHeading('Turn case files into a completed form.');
    const et1Cta = et1Section?.querySelector('.demo-copy .demo-cta');
    if (et1Cta) {
      et1Cta.textContent = 'Explore the founding pilot';
      et1Cta.setAttribute('href', `${pilotPath}?workflow=et1`);
    }
    const et1DemoLink = et1Section?.querySelector('.et1-book-demo');
    if (et1DemoLink) et1DemoLink.href = `${pilotPath}?workflow=et1#book`;

    const scheduleCta = byHeading('Know what the claim is worth.')?.querySelector('.demo-cta');
    if (scheduleCta) {
      scheduleCta.textContent = 'Test this with your workflow';
      scheduleCta.setAttribute('href', `${pilotPath}?workflow=schedule#book`);
    }

    const phoneSection = byHeading('Let Lylo answer the phone.');
    if (phoneSection) {
      const label = phoneSection.querySelector('.call-label');
      if (label) label.textContent = 'Try the live receptionist';
      phoneSection.querySelectorAll('.call-number').forEach(call => call.classList.add('phone-live-cta'));
      const extraCta = phoneSection.querySelector('.demo-copy > .demo-cta');
      if (extraCta) extraCta.remove();
      const callBox = phoneSection.querySelector('.call-box');
      if (callBox && !phoneSection.querySelector('.phone-call-helper')) {
        const helper = document.createElement('div');
        helper.className = 'phone-call-helper';
        helper.textContent = 'Call Lylo and speak as if you were a client.';
        callBox.appendChild(helper);
      }
    }

    const final = document.querySelector('.final .reveal');
    if (final) {
      const title = final.querySelector('h3');
      const copy = final.querySelector('p');
      const cta = final.querySelector('.cta');
      if (title) title.textContent = 'We are looking for a small number of law firms to pilot Lylo with us.';
      if (copy) copy.textContent = 'Start with a short demo. If it looks useful, test Lylo on one synthetic or properly anonymised matter and compare it with your normal workflow.';
      if (cta) {
        cta.textContent = 'Apply for the founding pilot';
        cta.setAttribute('href', `${pilotPath}#book`);
      }
    }
  };

  const injectStyles = () => {
    if (document.getElementById('preview-cta-styles')) return;
    const style = document.createElement('style');
    style.id = 'preview-cta-styles';
    style.textContent = `
      #sra-warning,#demos,#privacy{scroll-margin-top:84px}
      .hero .cta,.final .cta{position:relative;isolation:isolate;overflow:hidden;transition:transform .25s ease,box-shadow .25s ease,filter .25s ease}
      .hero .cta::before,.final .cta::before{content:"";position:absolute;z-index:-1;top:-120%;left:-38%;width:42%;height:340%;transform:rotate(24deg);background:linear-gradient(90deg,transparent,rgba(255,255,255,.11),transparent);transition:left .58s ease;pointer-events:none}
      .hero .cta:hover,.final .cta:hover{transform:translateY(-3px);filter:brightness(1.07);box-shadow:inset 0 1px 0 rgba(255,255,255,.12),0 18px 44px rgba(42,112,210,.22),0 0 28px rgba(99,168,255,.08)}
      .hero .cta:hover::before,.final .cta:hover::before{left:108%}
      .lylo-community{padding:100px var(--gutter);border-top:1px solid rgba(255,255,255,.055);background:linear-gradient(180deg,rgba(12,22,37,.26),rgba(8,16,29,0))}
      .lylo-community-inner{width:100%;max-width:1040px;margin:0 auto;display:grid;grid-template-columns:1fr 1fr;gap:18px}
      .lylo-community-card{min-height:390px;padding:34px;border:1px solid rgba(255,255,255,.075);border-radius:24px;background:linear-gradient(155deg,rgba(255,255,255,.028),rgba(255,255,255,.01));box-shadow:0 26px 70px rgba(0,0,0,.15);display:flex;flex-direction:column;align-items:flex-start;overflow:hidden;position:relative}
      .lylo-community-card::after{content:"";position:absolute;width:250px;height:250px;border-radius:50%;right:-130px;top:-130px;background:radial-gradient(circle,rgba(99,168,255,.075),transparent 70%);pointer-events:none}
      .lylo-team-card::after{background:radial-gradient(circle,rgba(106,217,176,.055),transparent 70%)}
      .lylo-community-kicker{margin-bottom:17px;color:#88a8cf;font-size:10px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;position:relative;z-index:1}
      .lylo-community-card h3{max-width:430px;margin:0 0 14px;font-family:'Cormorant Garamond',serif;font-size:39px;font-weight:400;line-height:1.03;letter-spacing:-.028em;color:#f5f7fa;position:relative;z-index:1}
      .lylo-community-card p{max-width:445px;margin:0;color:#96a5b8;font-size:13.5px;line-height:1.65;position:relative;z-index:1}
      .lylo-research-meta{margin-top:19px;color:#7589a3;font-size:10px;letter-spacing:.04em;position:relative;z-index:1}
      .lylo-team-mini{width:100%;display:grid;gap:9px;margin:0 0 20px;position:relative;z-index:1}
      .lylo-mini-person{display:grid;grid-template-columns:58px 1fr;gap:13px;align-items:center;padding:10px 12px;border:1px solid rgba(255,255,255,.06);border-radius:15px;background:rgba(255,255,255,.018)}
      .lylo-mini-person img{width:58px;height:58px;border-radius:50%;object-fit:cover;object-position:center;border:1px solid rgba(255,255,255,.10);box-shadow:0 8px 22px rgba(0,0,0,.2);background:#101a29}
      .lylo-mini-person strong{display:block;color:#eef4fb;font-size:12.5px;margin-bottom:4px}.lylo-mini-person span{display:block;color:#7f91a7;font-size:10.2px;line-height:1.4}
      .lylo-community-button{margin-top:auto;min-height:46px;padding:0 18px;display:inline-flex;align-items:center;justify-content:center;gap:10px;border-radius:999px;border:1px solid transparent;background:linear-gradient(180deg,rgba(18,34,55,.98),rgba(10,21,36,.98)) padding-box,linear-gradient(115deg,rgba(124,191,255,.64),rgba(111,221,183,.28),rgba(151,122,255,.44)) border-box;color:#f4f8fd;text-decoration:none;font-size:12px;font-weight:650;box-shadow:0 10px 30px rgba(42,112,210,.12);position:relative;z-index:1;transition:transform .2s ease,filter .2s ease}
      .lylo-community-button.secondary{background:rgba(19,36,58,.5);border:1px solid rgba(120,187,255,.18)}
      .lylo-community-button:hover{transform:translateY(-2px);filter:brightness(1.07)}
      .lylo-community-button span{font-size:15px}
      .lylo-community-note{margin-top:9px;color:#63758b;font-size:10px;line-height:1.45;position:relative;z-index:1}
      .phone-call-helper{margin-top:11px;color:#77879a;font-size:11px;line-height:1.45}
      .phone-section .phone-live-cta{position:relative;isolation:isolate;overflow:hidden;gap:12px;margin-top:0;min-height:50px;padding:0 21px;border-radius:999px;border:1px solid transparent;background:linear-gradient(180deg,rgba(16,29,48,.96),rgba(9,18,31,.98)) padding-box,linear-gradient(115deg,rgba(124,191,255,.72),rgba(111,221,183,.38),rgba(151,122,255,.52)) border-box;color:#f5f9ff!important;-webkit-text-fill-color:#f5f9ff!important;text-decoration:none;font-size:14px;font-weight:600;box-shadow:inset 0 1px 0 rgba(255,255,255,.09),0 12px 34px rgba(42,112,210,.13)}
      .phone-section .phone-live-cta::after{content:'→';font-size:17px}
      .desktop-nav{display:flex;align-items:center;gap:24px}.desktop-nav a{color:#aeb8c8;text-decoration:none;font-size:13px;font-weight:500}.desktop-nav a:hover{color:#fff}.desktop-nav .desktop-pilot-link{min-height:36px;padding:0 15px;display:inline-flex;align-items:center;border-radius:999px;color:#eef5fd;border:1px solid rgba(120,187,255,.22);background:rgba(19,36,58,.54)}
      @media(min-width:980px){.site-header .hamburger{display:none!important}}
      @media(max-width:979px){
        .desktop-nav{display:none!important}#mobileMenu .panel-inner a{font-size:21px;padding:17px 2px}#mobileMenu .panel-inner a:last-child{color:#eef5fd}
        .lylo-community{padding:64px 20px}.lylo-community-inner{grid-template-columns:1fr;gap:12px;max-width:390px}.lylo-community-card{min-height:0;padding:25px 21px;border-radius:18px;text-align:center;align-items:center}.lylo-community-card h3{font-size:33px;max-width:335px;margin-bottom:11px}.lylo-community-card p{font-size:12.5px;max-width:340px;line-height:1.58}.lylo-research-meta{margin-top:15px}.lylo-team-mini{max-width:340px}.lylo-mini-person{text-align:left}.lylo-community-button{margin-top:20px;width:100%;max-width:290px}.phone-call-helper{margin-top:10px;font-size:10.5px}.phone-section .phone-live-cta{min-height:48px;padding:0 20px;font-size:14px}
      }
      @media(prefers-reduced-motion:reduce){.hero .cta,.final .cta,.hero .cta::before,.final .cta::before{transition:none!important}.hero .cta:hover,.final .cta:hover{transform:none}}
    `;
    document.head.appendChild(style);
  };

  const init = () => {
    initSectionAnchors();
    initDesktopNav();
    initMobileNav();
    initCommunitySections();
    initDemoCtas();
    injectStyles();
  };

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init, {once:true});
  else init();
})();