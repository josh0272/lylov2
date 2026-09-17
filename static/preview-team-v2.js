(() => {
  const aboutPath = '/static/preview-about.html';

  const person = (image, initials, name, role) => `
    <div class="lylo-mini-person">
      <div class="lylo-mini-photo">
        <img src="${image}" alt="${name}" onerror="this.style.display='none';this.nextElementSibling.style.display='grid'">
        <span class="lylo-mini-fallback">${initials}</span>
      </div>
      <strong>${name}</strong>
      <span>${role}</span>
    </div>`;

  const render = () => {
    const research = document.querySelector('.lylo-research-card');
    const team = document.querySelector('.lylo-team-card');
    const community = document.querySelector('.lylo-community');
    const inner = document.querySelector('.lylo-community-inner');
    if (!research || !team || !community || !inner) return;

    community.classList.add('lylo-mini-sections');
    research.classList.add('lylo-mini-section', 'lylo-mini-research');
    team.classList.add('lylo-mini-section', 'lylo-mini-team');

    research.innerHTML = `
      <div class="lylo-mini-inner">
        <div class="lylo-mini-kicker">Help shape Lylo</div>
        <h3>Help us make Lylo better.</h3>
        <p>If you work or study in law, your answers will help us decide what to improve and what to build next.</p>
        <a class="lylo-mini-button" href="/research">Take the questionnaire <span>→</span></a>
      </div>
    `;

    team.innerHTML = `
      <div class="lylo-mini-inner">
        <div class="lylo-mini-kicker">Behind Lylo</div>
        <h3>Meet the people behind Lylo.</h3>
        <div class="lylo-mini-people">
          ${person('/static/joshua-profile.jpg', 'JS', 'Joshua Sam', 'Product & technology')}
          ${person('/static/jessica-profile.jpg', 'JJ', 'Jessica Jayan', 'Legal research & workflow')}
        </div>
        <a class="lylo-mini-button" href="${aboutPath}">Meet us <span>→</span></a>
      </div>
    `;

    let style = document.getElementById('preview-mini-section-styles');
    if (!style) {
      style = document.createElement('style');
      style.id = 'preview-mini-section-styles';
      style.textContent = `
        .lylo-mini-sections{padding:0!important;border-top:1px solid rgba(255,255,255,.055)!important;background:linear-gradient(180deg,rgba(11,20,34,.11),rgba(8,16,29,0))!important}
        .lylo-mini-sections .lylo-community-inner{display:block!important;width:100%!important;max-width:none!important;margin:0!important;padding:0!important}
        .lylo-mini-sections .lylo-community-card{display:block!important;width:100%!important;max-width:none!important;min-height:0!important;margin:0!important;padding:0!important;border:0!important;border-radius:0!important;background:none!important;box-shadow:none!important;text-align:center!important}
        .lylo-mini-section{padding:84px var(--gutter)!important}
        .lylo-mini-section+.lylo-mini-section{border-top:1px solid rgba(255,255,255,.055)!important}
        .lylo-mini-inner{width:100%;max-width:760px;margin:0 auto;text-align:center}
        .lylo-mini-kicker{margin-bottom:14px;color:#88a8cf;font-size:10px;font-weight:700;letter-spacing:.14em;text-transform:uppercase}
        .lylo-mini-section h3{max-width:680px!important;margin:0 auto 15px!important;font-family:'Cormorant Garamond',serif!important;font-size:clamp(40px,4.8vw,56px)!important;font-weight:400!important;line-height:1.03!important;letter-spacing:-.03em!important;color:#f5f7fa!important}
        .lylo-mini-section p{max-width:590px;margin:0 auto;color:#98a7b8;font-size:14px;line-height:1.68}
        .lylo-mini-button{display:inline-flex;align-items:center;justify-content:center;gap:10px;min-height:47px;margin-top:26px;padding:0 18px;border-radius:999px;border:1px solid transparent;background:linear-gradient(180deg,rgba(16,29,48,.96),rgba(9,18,31,.98)) padding-box,linear-gradient(115deg,rgba(124,191,255,.64),rgba(111,221,183,.30),rgba(151,122,255,.44)) border-box;color:#f5f9ff!important;text-decoration:none;font-size:12.5px;font-weight:600;box-shadow:inset 0 1px 0 rgba(255,255,255,.08),0 12px 30px rgba(42,112,210,.10);transition:transform .2s ease,filter .2s ease}
        .lylo-mini-button:hover{transform:translateY(-2px);filter:brightness(1.05)}
        .lylo-mini-button span{font-size:16px}
        .lylo-mini-people{display:flex;justify-content:center;gap:72px;margin:34px auto 0}
        .lylo-mini-person{width:180px;text-align:center}
        .lylo-mini-photo{position:relative;width:126px;height:126px;margin:0 auto 14px;border-radius:50%;overflow:hidden;border:1px solid rgba(255,255,255,.10);background:radial-gradient(circle at 35% 30%,rgba(98,157,226,.15),rgba(17,31,50,.82));box-shadow:0 16px 38px rgba(0,0,0,.22)}
        .lylo-mini-photo img{display:block;width:100%;height:100%;object-fit:cover}
        .lylo-mini-fallback{display:none;width:100%;height:100%;place-items:center;font-family:'Cormorant Garamond',serif;font-size:34px;color:#cdd9e7}
        .lylo-mini-person strong{display:block;margin-bottom:5px;color:#f2f6fb;font-family:'Cormorant Garamond',serif;font-size:25px;font-weight:400;line-height:1.05}
        .lylo-mini-person>span{display:block;color:#8092a8;font-size:10.5px;line-height:1.4}
        .lylo-mini-team .lylo-mini-button{margin-top:28px}
        @media(max-width:979px){
          .lylo-mini-section{padding:62px 20px!important}
          .lylo-mini-section h3{max-width:360px!important;font-size:38px!important;margin-bottom:13px!important}
          .lylo-mini-section p{max-width:345px;font-size:12.7px;line-height:1.62}
          .lylo-mini-button{width:100%;max-width:335px;margin-top:22px}
          .lylo-mini-people{gap:28px;margin-top:28px}
          .lylo-mini-person{width:138px}
          .lylo-mini-photo{width:104px;height:104px;margin-bottom:11px}
          .lylo-mini-person strong{font-size:22px}
          .lylo-mini-person>span{font-size:9.5px}
        }
        @media(max-width:390px){
          .lylo-mini-people{gap:16px}
          .lylo-mini-person{width:132px}
          .lylo-mini-photo{width:94px;height:94px}
        }
      `;
      document.head.appendChild(style);
    }
  };

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', () => setTimeout(render, 0), {once:true});
  else setTimeout(render, 0);
})();