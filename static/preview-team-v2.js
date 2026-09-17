(() => {
  const ABOUT_PATH = '/static/preview-about.html';
  const RESEARCH_PATH = '/research';

  const injectStyles = () => {
    if (document.getElementById('lylo-team-research-styles')) return;
    const style = document.createElement('style');
    style.id = 'lylo-team-research-styles';
    style.textContent = `
      .lylo-community{display:none!important}
      .lylo-people,.lylo-research-strip{border-top:1px solid rgba(255,255,255,.055)}
      .lylo-people{padding:86px var(--gutter) 80px}
      .lylo-people-inner{width:100%;max-width:940px;margin:0 auto;text-align:center}
      .lylo-section-kicker{margin-bottom:13px;color:#7893b3;font-size:10px;font-weight:700;letter-spacing:.14em;text-transform:uppercase}
      .lylo-people h3,.lylo-research-strip h3{font-family:'Cormorant Garamond',serif;font-weight:400;letter-spacing:-.03em;color:#f5f7fa}
      .lylo-people h3{font-size:clamp(36px,3.7vw,46px);line-height:1.05;margin:0 0 14px}
      .lylo-people-intro{max-width:610px;margin:0 auto;color:#8f9daf;font-size:14px;line-height:1.65}
      .lylo-people-list{display:flex;justify-content:center;gap:64px;margin:38px auto 29px}
      .lylo-person{display:flex;align-items:center;gap:18px;text-align:left;min-width:280px}
      .lylo-person img{width:90px;height:90px;flex:0 0 90px;display:block;border-radius:50%;object-fit:cover;object-position:center;border:1px solid rgba(255,255,255,.11);box-shadow:0 14px 34px rgba(0,0,0,.24);background:#101a29}
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
      @media(max-width:720px){
        .lylo-people{padding:68px 20px 64px}.lylo-people h3{font-size:36px}.lylo-people-intro{font-size:13px;max-width:340px}
        .lylo-people-list{gap:25px;margin-top:31px;flex-direction:column;align-items:center}.lylo-person{min-width:0;width:100%;max-width:330px;justify-content:flex-start;gap:16px}.lylo-person img{width:78px;height:78px;flex-basis:78px}
        .lylo-research-strip{padding:62px 20px}.lylo-research-strip h3{font-size:36px}.lylo-research-strip p{font-size:13px;max-width:350px}
      }
    `;
    document.head.appendChild(style);
  };

  const buildSections = () => {
    document.querySelector('.lylo-community')?.remove();
    document.querySelector('.lylo-people')?.remove();
    document.querySelector('.lylo-research-strip')?.remove();

    const finalSection = document.querySelector('.final');
    if (!finalSection) return;

    const people = document.createElement('section');
    people.className = 'lylo-people';
    people.innerHTML = `
      <div class="lylo-people-inner reveal">
        <div class="lylo-section-kicker">The people behind Lylo</div>
        <h3>Built by Joshua and Jessica.</h3>
        <p class="lylo-people-intro">A small team combining engineering and legal experience while Lylo is still being researched, tested and shaped.</p>
        <div class="lylo-people-list" aria-label="People behind Lylo">
          <div class="lylo-person">
            <img src="/static/joshua-profile.jpg" alt="Joshua Sam">
            <div><strong>Joshua Sam</strong><span class="role">Engineering · product & technology</span><span class="credential">MEng Electrical & Mechanical Engineering · University of Strathclyde</span></div>
          </div>
          <div class="lylo-person">
            <img src="/static/jessica-profile.jpg" alt="Jessica Jayan">
            <div><strong>Jessica Jayan</strong><span class="role">Legal research · workflow input</span><span class="credential">Scots (Clinical) LLB · DPLP · University of Strathclyde</span></div>
          </div>
        </div>
        <a class="lylo-secondary-link" href="${ABOUT_PATH}">Meet the people behind Lylo <span class="arrow">→</span></a>
      </div>`;

    const research = document.createElement('section');
    research.className = 'lylo-research-strip';
    research.innerHTML = `
      <div class="lylo-research-inner reveal">
        <div class="lylo-section-kicker">Research</div>
        <h3>Help us build Lylo around real legal work.</h3>
        <p>We are speaking with people in law about the work that takes the most time, where AI could genuinely help, and what firms would need before trusting it. Our short questionnaire helps shape what Lylo should become.</p>
        <div class="lylo-research-action"><a class="lylo-secondary-link" href="${RESEARCH_PATH}">Take the questionnaire <span class="arrow">→</span></a></div>
        <div class="lylo-research-note">Around 5 minutes · used for product research.</div>
      </div>`;

    finalSection.parentNode.insertBefore(people, finalSection);
    finalSection.parentNode.insertBefore(research, finalSection);

    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      people.querySelectorAll('.reveal').forEach(el => el.classList.add('in'));
      research.querySelectorAll('.reveal').forEach(el => el.classList.add('in'));
    } else if ('IntersectionObserver' in window) {
      const observer = new IntersectionObserver(entries => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            entry.target.classList.add('in');
            observer.unobserve(entry.target);
          }
        });
      }, { threshold: .12 });
      people.querySelectorAll('.reveal').forEach(el => observer.observe(el));
      research.querySelectorAll('.reveal').forEach(el => observer.observe(el));
    } else {
      people.querySelectorAll('.reveal').forEach(el => el.classList.add('in'));
      research.querySelectorAll('.reveal').forEach(el => el.classList.add('in'));
    }
  };

  const init = () => {
    injectStyles();
    buildSections();
  };

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init, { once:true });
  else init();
})();