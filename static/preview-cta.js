(() => {
  const text = (selector, value) => {
    const node = document.querySelector(selector);
    if (node) node.textContent = value;
    return node;
  };

  const pilotPath = '/preview/founding-pilot';
  const aboutPath = '/static/preview-about.html';
  const joshPhoto = 'data:image/jpeg;base64,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';
  const jessPhoto = 'data:image/jpeg;base64,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';

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
    nav.innerHTML = `<a href="#sra-warning">SRA warning</a><a href="#demos">Demos</a><a href="#privacy">Private AI</a><a class="desktop-pilot-link" href="${pilotPath}">Founding Pilot</a>`;
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
    panel.innerHTML = `<a href="#sra-warning">SRA warning</a><a href="#demos">Demos</a><a href="#privacy">Private AI</a><a href="${pilotPath}">Founding Pilot</a>`;
    panel.querySelectorAll('a').forEach((link) => link.addEventListener('click', closeMobileMenu));
  };

  const initCommunitySections = () => {
    const final = document.querySelector('.final');
    if (!final || document.querySelector('.lylo-community')) return;
    const section = document.createElement('section');
    section.className = 'lylo-community';
    section.innerHTML = `
      <div class="lylo-community-inner">
        <article class="lylo-community-card lylo-research-card">
          <div class="lylo-community-kicker">Help shape Lylo</div>
          <h3>Are you a solicitor, trainee or law student?</h3>
          <p>Tell us which legal tasks take too long, where AI could help and where it should stay out of the way.</p>
          <a class="lylo-community-link" href="/research">Share your experience <span>→</span></a>
          <div class="lylo-community-note">Short research questionnaire · no sales pitch.</div>
        </article>
        <article class="lylo-community-card lylo-team-card">
          <div class="lylo-community-kicker">The people behind Lylo</div>
          <div class="lylo-team-mini">
            <div class="lylo-mini-person"><img src="${joshPhoto}" alt="Joshua Sam"><div><strong>Joshua Sam</strong><span>Engineering · product & technology</span></div></div>
            <div class="lylo-mini-person"><img src="${jessPhoto}" alt="Jessica Jayan"><div><strong>Jessica Jayan</strong><span>Scots (Clinical) LLB 2:1 · DPLP · Strathclyde Law Clinic</span></div></div>
          </div>
          <p>Engineering builds the product. Legal experience helps us test whether it actually fits the way firms work.</p>
          <a class="lylo-community-link" href="${aboutPath}">Meet the team <span>→</span></a>
          <div class="lylo-community-note">Backgrounds, experience and CVs.</div>
        </article>
      </div>`;
    final.parentNode.insertBefore(section, final);
  };

  const init = () => {
    initSectionAnchors();
    initDesktopNav();
    initMobileNav();
    initCommunitySections();

    const heroCta = document.querySelector('.hero .cta');
    if (heroCta) { heroCta.textContent = 'Book a 20-minute demo'; heroCta.setAttribute('href', `${pilotPath}#book`); }
    text('.hero .note', 'See Lylo, ask questions, and decide if the pilot is worth testing.');

    const sections = Array.from(document.querySelectorAll('.demo-section'));
    const byHeading = (heading) => sections.find((section) => section.querySelector('.demo-copy h3')?.textContent.trim() === heading);
    const caseSection = byHeading('Ask the case. Get the answer.');
    const caseCta = caseSection?.querySelector('.demo-cta');
    if (caseCta) { caseCta.textContent = 'Explore the founding pilot'; caseCta.setAttribute('href', pilotPath); }
    const et1Section = byHeading('Turn case files into a completed form.');
    const et1Cta = et1Section?.querySelector('.demo-copy .demo-cta');
    if (et1Cta) { et1Cta.textContent = 'Explore the founding pilot'; et1Cta.setAttribute('href', `${pilotPath}?workflow=et1`); }
    const et1Form = et1Section?.querySelector('#et1-form-suggest');
    const et1Row = et1Form?.querySelector('.et1-suggest-row');
    if (et1Row && !et1Row.querySelector('.et1-book-demo')) {
      const demoLink = document.createElement('a'); demoLink.className='et1-book-demo'; demoLink.href=`${pilotPath}?workflow=et1#book`; demoLink.textContent='Book a 20-minute demo'; et1Row.appendChild(demoLink);
    } else { const demoLink = et1Row?.querySelector('.et1-book-demo'); if (demoLink) demoLink.href=`${pilotPath}?workflow=et1#book`; }
    const scheduleSection = byHeading('Know what the claim is worth.');
    const scheduleCta = scheduleSection?.querySelector('.demo-cta');
    if (scheduleCta) { scheduleCta.textContent='Test this with your workflow'; scheduleCta.setAttribute('href', `${pilotPath}?workflow=schedule#book`); }
    const phoneSection = byHeading('Let Lylo answer the phone.');
    if (phoneSection) {
      const label = phoneSection.querySelector('.call-label'); if (label) label.textContent='Try the live receptionist';
      phoneSection.querySelectorAll('.call-number').forEach((call)=>call.classList.add('phone-live-cta'));
      const extraCta = phoneSection.querySelector('.demo-copy > .demo-cta'); if (extraCta) extraCta.remove();
      const helper = document.createElement('div'); helper.className='phone-call-helper'; helper.textContent='Call Lylo and speak as if you were a client.';
      const callBox = phoneSection.querySelector('.call-box'); if (callBox && !phoneSection.querySelector('.phone-call-helper')) callBox.appendChild(helper);
    }
    const final = document.querySelector('.final .reveal');
    if (final) {
      const title=final.querySelector('h3'), copy=final.querySelector('p'), cta=final.querySelector('.cta');
      if (title) title.textContent='We are looking for a small number of law firms to pilot Lylo with us.';
      if (copy) copy.textContent='Start with a short demo. If it looks useful, test Lylo on one synthetic or properly anonymised matter and compare it with your normal workflow.';
      if (cta) { cta.textContent='Apply for the founding pilot'; cta.setAttribute('href', `${pilotPath}#book`); }
    }

    if (!document.getElementById('preview-cta-styles')) {
      const style=document.createElement('style'); style.id='preview-cta-styles'; style.textContent=`
        #sra-warning,#demos,#privacy{scroll-margin-top:84px}
        .lylo-community{padding:98px var(--gutter);border-top:1px solid rgba(255,255,255,.055);background:linear-gradient(180deg,rgba(9,17,29,.18),rgba(8,16,29,0))}
        .lylo-community-inner{width:100%;max-width:1040px;margin:0 auto;display:grid;grid-template-columns:1fr 1fr;gap:16px}
        .lylo-community-card{min-height:330px;padding:34px 32px;border:1px solid rgba(255,255,255,.075);border-radius:22px;background:linear-gradient(155deg,rgba(255,255,255,.026),rgba(255,255,255,.011));box-shadow:0 22px 60px rgba(0,0,0,.13);display:flex;flex-direction:column;align-items:flex-start}
        .lylo-community-kicker{margin-bottom:17px;color:#88a8cf;font-size:10px;font-weight:700;letter-spacing:.14em;text-transform:uppercase}
        .lylo-community-card h3{max-width:430px;margin:0 0 14px;font-family:'Cormorant Garamond',serif;font-size:36px;font-weight:400;line-height:1.06;letter-spacing:-.025em;color:#f5f7fa}
        .lylo-community-card p{max-width:450px;margin:0;color:#96a5b8;font-size:13.5px;line-height:1.65}
        .lylo-team-mini{width:100%;display:grid;gap:10px;margin:0 0 18px}
        .lylo-mini-person{display:grid;grid-template-columns:50px 1fr;gap:12px;align-items:center;padding:9px 10px;border:1px solid rgba(255,255,255,.06);border-radius:14px;background:rgba(255,255,255,.015)}
        .lylo-mini-person img{width:50px;height:50px;border-radius:50%;object-fit:cover;border:1px solid rgba(255,255,255,.10)}
        .lylo-mini-person strong{display:block;color:#eef4fb;font-size:12px;margin-bottom:3px}.lylo-mini-person span{display:block;color:#7f91a7;font-size:10px;line-height:1.35}
        .lylo-community-link{margin-top:auto;padding-top:22px;display:inline-flex;align-items:center;gap:9px;color:#edf4fc;text-decoration:none;font-size:12px;font-weight:600}
        .lylo-community-link span{font-size:15px;transition:transform .2s ease}.lylo-community-link:hover span{transform:translateX(3px)}
        .lylo-community-note{margin-top:8px;color:#63758b;font-size:10px;line-height:1.45}
        .phone-call-helper{margin-top:11px;color:#77879a;font-size:11px;line-height:1.45}
        .phone-section .phone-live-cta{position:relative;isolation:isolate;overflow:hidden;gap:12px;margin-top:0;min-height:50px;padding:0 21px;border-radius:999px;border:1px solid transparent;background:linear-gradient(180deg,rgba(16,29,48,.96),rgba(9,18,31,.98)) padding-box,linear-gradient(115deg,rgba(124,191,255,.72),rgba(111,221,183,.38),rgba(151,122,255,.52)) border-box;color:#f5f9ff!important;-webkit-text-fill-color:#f5f9ff!important;text-decoration:none;font-size:14px;font-weight:600;box-shadow:inset 0 1px 0 rgba(255,255,255,.09),0 12px 34px rgba(42,112,210,.13)}
        .phone-section .phone-live-cta::after{content:'→';font-size:17px}.desktop-nav{display:flex;align-items:center;gap:24px}.desktop-nav a{color:#aeb8c8;text-decoration:none;font-size:13px;font-weight:500}.desktop-nav a:hover{color:#fff}.desktop-nav .desktop-pilot-link{min-height:36px;padding:0 15px;display:inline-flex;align-items:center;border-radius:999px;color:#eef5fd;border:1px solid rgba(120,187,255,.22);background:rgba(19,36,58,.54)}
        @media(min-width:980px){.site-header .hamburger{display:none!important}}
        @media(max-width:979px){.desktop-nav{display:none!important}#mobileMenu .panel-inner a{font-size:21px;padding:17px 2px}#mobileMenu .panel-inner a:last-child{color:#eef5fd}.lylo-community{padding:64px 20px}.lylo-community-inner{grid-template-columns:1fr;gap:11px;max-width:390px}.lylo-community-card{min-height:0;padding:24px 20px;border-radius:17px;text-align:center;align-items:center}.lylo-community-kicker{margin-bottom:12px}.lylo-community-card h3{font-size:32px;max-width:330px;margin-bottom:10px}.lylo-community-card p{font-size:12.5px;max-width:335px;line-height:1.58}.lylo-team-mini{max-width:335px}.lylo-mini-person{text-align:left}.lylo-community-link{margin-top:18px;padding-top:0}.phone-call-helper{margin-top:10px;font-size:10.5px}.phone-section .phone-live-cta{min-height:48px;padding:0 20px;font-size:14px}}
      `; document.head.appendChild(style);
    }
  };

  if (document.readyState==='loading') document.addEventListener('DOMContentLoaded',init,{once:true}); else init();
})();