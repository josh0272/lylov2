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

  const initTeamResearch = () => {
    const finalSection = document.querySelector('.final');
    if (!finalSection || document.querySelector('.lylo-people')) return;

    const aboutPath = 'static/preview-about.html';
    const joshPhoto = 'static/joshua-profile.jpg';
    const jessPhoto = 'static/jessica-profile.jpg';

    const people = document.createElement('section');
    people.className = 'lylo-people';
    people.innerHTML = `
      <div class="lylo-people-inner reveal in">
        <div class="lylo-section-kicker">The people behind Lylo</div>
        <h3>Built by Joshua and Jessica.</h3>
        <p class="lylo-people-intro">A small team combining engineering and legal experience while Lylo is being researched, tested and shaped.</p>
        <div class="lylo-people-list" aria-label="People behind Lylo">
          <div class="lylo-person">
            <a class="lylo-person-photo" href="${aboutPath}#joshua" aria-label="Read more about Joshua Sam"><img src="${joshPhoto}" alt="Joshua Sam"></a>
            <div><strong>Joshua Sam</strong><span class="role">Product & engineering</span><span class="credential">MEng Electrical & Mechanical Engineering · University of Strathclyde</span></div>
          </div>
          <div class="lylo-person">
            <a class="lylo-person-photo" href="${aboutPath}#jessica" aria-label="Read more about Jessica Jayan"><img src="${jessPhoto}" alt="Jessica Jayan"></a>
            <div><strong>Jessica Jayan</strong><span class="role">Legal research & workflow</span><span class="credential">Scots (Clinical) LLB · DPLP · University of Strathclyde</span></div>
          </div>
        </div>
        <a class="lylo-secondary-link" href="${aboutPath}">Meet the people behind Lylo <span class="arrow">→</span></a>
      </div>`;

    const research = document.createElement('section');
    research.className = 'lylo-research-strip';
    research.innerHTML = `
      <div class="lylo-research-inner reveal in">
        <div class="lylo-section-kicker">Research</div>
        <h3>Help us build Lylo around real legal work.</h3>
        <p>We are speaking with people in law about the work that takes the most time, where AI could genuinely help, and what firms would need before trusting it. Our short questionnaire helps shape what Lylo should become.</p>
        <div class="lylo-research-action"><a class="lylo-secondary-link" href="/research">Take the questionnaire <span class="arrow">→</span></a></div>
        <div class="lylo-research-note">Around 5 minutes · used for product research.</div>
      </div>`;

    finalSection.parentNode.insertBefore(people, finalSection);
    finalSection.parentNode.insertBefore(research, finalSection);
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

    initTeamResearch();

    if (!document.getElementById('preview-cta-styles')) {
      const style = document.createElement('style');
      style.id = 'preview-cta-styles';
      style.textContent = `
        #sra-warning,#demos,#privacy{scroll-margin-top:84px}
        .beyond-casework{padding:82px var(--gutter) 18px;text-align:center;border-top:1px solid rgba(255,255,255,.055);background:linear-gradient(180deg,rgba(11,20,34,.12),rgba(8,16,29,0))}
        .beyond-casework-inner{width:100%;max-width:720px;margin:0 auto}
        .beyond-kicker,.lylo-section-kicker{margin-bottom:13px;color:#88a8cf;font-size:10px;font-weight:700;letter-spacing:.14em;text-transform:uppercase}
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

        .lylo-people,.lylo-research-strip{border-top:1px solid rgba(255,255,255,.055)}
        .lylo-people{padding:86px var(--gutter) 80px}
        .lylo-people-inner{width:100%;max-width:940px;margin:0 auto;text-align:center}
        .lylo-people h3,.lylo-research-strip h3{font-family:'Cormorant Garamond',serif;font-weight:400;letter-spacing:-.03em;color:#f5f7fa}
        .lylo-people h3{font-size:clamp(36px,3.7vw,46px);line-height:1.05;margin:0 0 14px}
        .lylo-people-intro{max-width:610px;margin:0 auto;color:#8f9daf;font-size:14px;line-height:1.65}
        .lylo-people-list{display:flex;justify-content:center;gap:64px;margin:38px auto 29px}
        .lylo-person{display:flex;align-items:center;gap:18px;text-align:left;min-width:280px}
        .lylo-person-photo{display:block;border-radius:50%;transition:transform .2s ease,filter .2s ease}
        .lylo-person-photo:hover{transform:translateY(-2px);filter:brightness(1.06)}
        .lylo-person img{width:90px;height:90px;display:block;border-radius:50%;object-fit:cover;object-position:center;border:1px solid rgba(255,255,255,.11);box-shadow:0 14px 34px rgba(0,0,0,.24);background:#101a29}
        .lylo-person strong{display:block;color:#edf3fa;font-size:14.5px;font-weight:600;margin-bottom:5px}
        .lylo-person .role{display:block;color:#75869b;font-size:11px;line-height:1.45;max-width:205px}
        .lylo-person .credential{display:block;margin-top:4px;color:#9fb0c4;font-size:10.5px;line-height:1.42;max-width:225px}
        .lylo-secondary-link{display:inline-flex;align-items:center;gap:8px;color:#9db4cf;text-decoration:none;font-size:12px;font-weight:600;border-bottom:1px solid rgba(157,180,207,.20);padding-bottom:3px;transition:color .2s ease,border-color .2s ease}
        .lylo-secondary-link:hover{color:#e7f0fb;border-color:rgba(231,240,251,.42)}
        .lylo-secondary-link .arrow{font-size:14px;transition:transform .2s ease}.lylo-secondary-link:hover .arrow{transform:translateX(3px)}
        .lylo-research-strip{padding:76px var(--gutter);background:linear-gradient(180deg,rgba(255,255,255,.008),rgba(255,255,255,0))}
        .lylo-research-inner{width:100%;max-width:740px;margin:0 auto;text-align:center}
        .lylo-research-strip h3{font-size:clamp(36px,3.7vw,46px);line-height:1.06;margin:0 0 16px}
        .lylo-research-strip p{max-width:620px;margin:0 auto;color:#8f9daf;font-size:14px;line-height:1.68}
        .lylo-research-action{margin-top:23px}
        .lylo-research-note{margin-top:10px;color:#607187;font-size:10px;line-height:1.45}

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
          .lylo-people{padding:68px 20px 64px}.lylo-people h3{font-size:36px}.lylo-people-intro{font-size:13px;max-width:340px}
          .lylo-people-list{gap:25px;margin-top:31px;flex-direction:column;align-items:center}.lylo-person{min-width:0;width:100%;max-width:330px;justify-content:flex-start;gap:16px}.lylo-person img{width:78px;height:78px}
          .lylo-research-strip{padding:62px 20px}.lylo-research-strip h3{font-size:36px}.lylo-research-strip p{font-size:13px;max-width:350px}
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