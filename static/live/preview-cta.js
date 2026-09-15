(() => {
  const text = (selector, value) => {
    const node = document.querySelector(selector);
    if (node) node.textContent = value;
    return node;
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
      <a class="desktop-pilot-link" href="/founding-pilot">Founding Pilot</a>
    `;
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
      <a href="/founding-pilot">Founding Pilot</a>
    `;

    panel.querySelectorAll('a').forEach((link) => {
      link.addEventListener('click', closeMobileMenu);
    });
  };

  const init = () => {
    initSectionAnchors();
    initDesktopNav();
    initMobileNav();

    const heroCta = document.querySelector('.hero .cta');
    if (heroCta) {
      heroCta.textContent = 'Book a 20-minute demo';
      heroCta.setAttribute('href', '/founding-pilot#book');
    }
    text('.hero .note', 'See Lylo, ask questions, and decide if the pilot is worth testing.');

    const sections = Array.from(document.querySelectorAll('.demo-section'));
    const byHeading = (heading) => sections.find((section) => section.querySelector('.demo-copy h3')?.textContent.trim() === heading);

    const caseSection = byHeading('Ask the case. Get the answer.');
    const caseCta = caseSection?.querySelector('.demo-cta');
    if (caseCta) {
      caseCta.textContent = 'Explore the founding pilot';
      caseCta.setAttribute('href', '/founding-pilot');
    }

    const et1Section = byHeading('Turn case files into a completed form.');
    const et1Cta = et1Section?.querySelector('.demo-copy .demo-cta');
    if (et1Cta) {
      et1Cta.textContent = 'Explore the founding pilot';
      et1Cta.setAttribute('href', '/founding-pilot?workflow=et1');
    }

    const et1Form = et1Section?.querySelector('#et1-form-suggest');
    const et1Row = et1Form?.querySelector('.et1-suggest-row');
    if (et1Row && !et1Row.querySelector('.et1-book-demo')) {
      const demoLink = document.createElement('a');
      demoLink.className = 'et1-book-demo';
      demoLink.href = '/founding-pilot?workflow=et1#book';
      demoLink.textContent = 'Book a 20-minute demo';
      et1Row.appendChild(demoLink);
    } else {
      const demoLink = et1Row?.querySelector('.et1-book-demo');
      if (demoLink) demoLink.href = '/founding-pilot?workflow=et1#book';
    }

    const scheduleSection = byHeading('Know what the claim is worth.');
    const scheduleCta = scheduleSection?.querySelector('.demo-cta');
    if (scheduleCta) {
      scheduleCta.textContent = 'Test this with your workflow';
      scheduleCta.setAttribute('href', '/founding-pilot?workflow=schedule#book');
    }

    const phoneSection = byHeading('Let Lylo answer the phone.');
    if (phoneSection) {
      if (!document.querySelector('.beyond-casework')) {
        const bridge = document.createElement('section');
        bridge.className = 'beyond-casework';
        bridge.innerHTML = `
          <div class="beyond-casework-inner">
            <div class="beyond-kicker">Beyond casework</div>
            <h3>AI can help with more than legal documents.</h3>
            <p>Alongside private legal workflows, Lylo can also be adapted for useful firm-wide tasks. One example is out-of-hours call handling.</p>
            <div class="beyond-boundary">The voice receptionist uses separate cloud voice services and is separate from Lylo’s private on-premise legal AI.</div>
          </div>
        `;
        phoneSection.parentNode.insertBefore(bridge, phoneSection);
      }

      const label = phoneSection.querySelector('.call-label');
      if (label) label.textContent = 'Try the live receptionist';

      phoneSection.querySelectorAll('.call-number').forEach((call) => {
        call.classList.add('phone-live-cta');
      });

      const extraCta = phoneSection.querySelector('.demo-copy > .demo-cta');
      if (extraCta) extraCta.remove();

      const helper = document.createElement('div');
      helper.className = 'phone-call-helper';
      helper.textContent = 'Call Lylo and speak as if you were a client.';
      const callBox = phoneSection.querySelector('.call-box');
      if (callBox && !phoneSection.querySelector('.phone-call-helper')) callBox.appendChild(helper);
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
        cta.setAttribute('href', '/founding-pilot#book');
      }
    }

    if (!document.getElementById('preview-cta-styles')) {
      const style = document.createElement('style');
      style.id = 'preview-cta-styles';
      style.textContent = `
        #sra-warning,#demos,#privacy{scroll-margin-top:84px}
        .beyond-casework{padding:82px var(--gutter) 18px;text-align:center;border-top:1px solid rgba(255,255,255,.055);background:linear-gradient(180deg,rgba(11,20,34,.12),rgba(8,16,29,0))}
        .beyond-casework-inner{width:100%;max-width:720px;margin:0 auto}
        .beyond-kicker{margin-bottom:13px;color:#88a8cf;font-size:10px;font-weight:700;letter-spacing:.14em;text-transform:uppercase}
        .beyond-casework h3{max-width:680px;margin:0 auto 15px;font-family:'Cormorant Garamond',serif;font-size:clamp(38px,4.5vw,54px);font-weight:400;line-height:1.04;letter-spacing:-.03em;color:#f5f7fa}
        .beyond-casework p{max-width:610px;margin:0 auto;color:#9eabba;font-size:15px;line-height:1.65}
        .beyond-boundary{max-width:610px;margin:17px auto 0;padding-top:15px;border-top:1px solid rgba(255,255,255,.055);color:#68798e;font-size:10.5px;line-height:1.55}
        .beyond-casework + .phone-section{border-top:0;padding-top:82px}
        .phone-call-helper{margin-top:11px;color:#77879a;font-size:11px;line-height:1.45}
        .phone-section .phone-live-cta{position:relative;isolation:isolate;overflow:hidden;gap:12px;margin-top:0;min-height:50px;padding:0 21px;border-radius:999px;border:1px solid transparent;background:linear-gradient(180deg,rgba(16,29,48,.96),rgba(9,18,31,.98)) padding-box,linear-gradient(115deg,rgba(124,191,255,.72),rgba(111,221,183,.38),rgba(151,122,255,.52)) border-box;color:#f5f9ff!important;-webkit-text-fill-color:#f5f9ff!important;text-decoration:none;font-size:14px;font-weight:600;letter-spacing:.005em;box-shadow:inset 0 1px 0 rgba(255,255,255,.09),0 12px 34px rgba(42,112,210,.13),0 0 0 1px rgba(255,255,255,.018);transition:transform .25s ease,box-shadow .25s ease,filter .25s ease}
        .phone-section .phone-live-cta::after{content:"→";font-size:17px;line-height:1;transition:transform .25s ease}
        .phone-section .phone-live-cta::before{content:"";position:absolute;z-index:-1;top:-120%;left:-35%;width:42%;height:340%;transform:rotate(24deg);background:linear-gradient(90deg,transparent,rgba(255,255,255,.10),transparent);transition:left .55s ease;pointer-events:none}
        .phone-section .phone-live-cta:hover{transform:translateY(-3px);filter:brightness(1.06);border-color:transparent;background:linear-gradient(180deg,rgba(16,29,48,.96),rgba(9,18,31,.98)) padding-box,linear-gradient(115deg,rgba(124,191,255,.72),rgba(111,221,183,.38),rgba(151,122,255,.52)) border-box;box-shadow:inset 0 1px 0 rgba(255,255,255,.12),0 18px 46px rgba(42,112,210,.22),0 0 26px rgba(99,168,255,.09)}
        .phone-section .phone-live-cta:hover::after{transform:translateX(4px)}
        .phone-section .phone-live-cta:hover::before{left:105%}
        .et1-book-demo{display:none;height:44px;padding:0 17px;border:1px solid rgba(120,187,255,.28);border-radius:13px;background:rgba(31,58,92,.92);color:#f2f7fd;font:600 12px/1 inherit;text-decoration:none;align-items:center;justify-content:center;white-space:nowrap;box-shadow:0 8px 22px rgba(34,94,162,.12);transition:transform .18s ease,border-color .18s ease,background .18s ease,opacity .22s ease}
        .et1-book-demo:hover{transform:translateY(-1px);border-color:rgba(127,184,255,.38);background:rgba(38,67,104,.98)}
        .et1-suggest.is-submitted .et1-book-demo{display:inline-flex;animation:et1BookReveal .26s ease both}
        @keyframes et1BookReveal{from{opacity:0;transform:translateX(7px)}to{opacity:1;transform:translateX(0)}}
        .desktop-nav{display:flex;align-items:center;gap:24px}
        .desktop-nav a{color:#aeb8c8;text-decoration:none;font-size:13px;font-weight:500;line-height:1;transition:color .2s ease,background .2s ease,border-color .2s ease,transform .2s ease}
        .desktop-nav a:hover{color:#fff}
        .desktop-nav .desktop-pilot-link{min-height:36px;padding:0 15px;display:inline-flex;align-items:center;border-radius:999px;color:#eef5fd;border:1px solid rgba(120,187,255,.22);background:rgba(19,36,58,.54);box-shadow:inset 0 1px 0 rgba(255,255,255,.035)}
        .desktop-nav .desktop-pilot-link:hover{transform:translateY(-1px);border-color:rgba(127,184,255,.34);background:rgba(24,45,72,.74)}
        @media(min-width:980px){.site-header .hamburger{display:none!important}}
        @media(max-width:979px){
          .desktop-nav{display:none!important}
          #mobileMenu .panel-inner a{font-size:21px;padding:17px 2px}
          #mobileMenu .panel-inner a:last-child{color:#eef5fd}
          .beyond-casework{padding:62px 20px 4px}
          .beyond-casework h3{max-width:340px;font-size:37px;margin-bottom:12px}
          .beyond-casework p{max-width:345px;font-size:13px;line-height:1.58}
          .beyond-boundary{max-width:345px;margin-top:14px;padding-top:13px;font-size:9.5px}
          .beyond-casework + .phone-section{padding-top:62px}
          .phone-call-helper{margin-top:10px;font-size:10.5px}
          .phone-section .phone-live-cta{min-height:48px;padding:0 20px;font-size:14px}
          .et1-book-demo{height:41px;padding:0 12px;border-radius:11px;font-size:11px}
          .et1-suggest.is-submitted .et1-suggest-row{align-items:center}
          .et1-suggest.is-submitted button,.et1-suggest.is-submitted .et1-book-demo{flex:1 1 0}
        }
        @media(max-width:430px){
          .et1-suggest.is-submitted .et1-suggest-row{flex-direction:column}
          .et1-suggest.is-submitted button,.et1-suggest.is-submitted .et1-book-demo{width:100%}
        }
      `;
      document.head.appendChild(style);
    }
  };

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init, { once:true });
  else init();
})();