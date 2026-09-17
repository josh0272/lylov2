(() => {
  const aboutPath = '/static/preview-about.html';

  const person = (image, initials, name, role, detail) => `
    <div class="lylo-card-person">
      <div class="lylo-card-photo">
        <img src="${image}" alt="${name}" onerror="this.style.display='none';this.nextElementSibling.style.display='grid'">
        <span class="lylo-card-fallback">${initials}</span>
      </div>
      <div class="lylo-card-person-copy">
        <strong>${name}</strong>
        <span>${role}</span>
        <small>${detail}</small>
      </div>
    </div>`;

  const render = () => {
    const research = document.querySelector('.lylo-research-card');
    const team = document.querySelector('.lylo-team-card');
    if (!research || !team) return;

    research.innerHTML = `
      <div class="lylo-card-kicker">Research</div>
      <h3>Help us make Lylo better.</h3>
      <p class="lylo-card-copy">If you work or study in law, tell us where work slows down, what you would trust AI to help with, and what Lylo should improve next.</p>
      <div class="lylo-card-meta">
        <span>Current research</span>
        <strong>ET1s · Schedules of Loss · legal AI</strong>
      </div>
      <a class="lylo-card-button" href="/research">Take the questionnaire <span>→</span></a>
    `;

    team.innerHTML = `
      <div class="lylo-card-kicker">About us</div>
      <h3>Joshua Sam & Jessica Jayan</h3>
      <div class="lylo-card-people">
        ${person('/static/joshua-profile.jpg', 'JS', 'Joshua Sam', 'Product & technology', 'MEng · University of Strathclyde')}
        ${person('/static/jessica-profile.jpg', 'JJ', 'Jessica Jayan', 'Legal research & workflow', 'LLB · DPLP · Strathclyde Law Clinic')}
      </div>
      <a class="lylo-card-button" href="${aboutPath}">Meet the people behind Lylo <span>→</span></a>
    `;

    let style = document.getElementById('preview-team-reset-styles');
    if (!style) {
      style = document.createElement('style');
      style.id = 'preview-team-reset-styles';
      style.textContent = `
        .lylo-community{padding-top:92px!important;padding-bottom:92px!important}
        .lylo-community-inner{align-items:stretch!important;gap:18px!important}
        .lylo-community-card{min-height:405px!important;padding:44px 42px!important;border-radius:22px!important;border:1px solid rgba(255,255,255,.065)!important;background:linear-gradient(155deg,rgba(255,255,255,.022),rgba(255,255,255,.009))!important;box-shadow:0 22px 55px rgba(0,0,0,.10)!important}
        .lylo-card-kicker{margin-bottom:22px;color:#88a8cf;font-size:10px;font-weight:700;letter-spacing:.14em;text-transform:uppercase}
        .lylo-community-card h3{max-width:440px!important;margin:0 0 16px!important;font-family:'Cormorant Garamond',serif!important;font-size:41px!important;font-weight:400!important;line-height:1.03!important;letter-spacing:-.025em!important;color:#f5f7fa!important}
        .lylo-card-copy{max-width:430px;margin:0;color:#96a5b8;font-size:13.5px;line-height:1.68}
        .lylo-card-meta{margin-top:31px;padding-top:17px;border-top:1px solid rgba(255,255,255,.06)}
        .lylo-card-meta span{display:block;margin-bottom:5px;color:#61758d;font-size:9px;font-weight:700;letter-spacing:.11em;text-transform:uppercase}
        .lylo-card-meta strong{display:block;color:#becbd9;font-size:11.5px;font-weight:500;line-height:1.5}
        .lylo-card-people{display:grid;grid-template-columns:1fr 1fr;gap:22px;margin:8px 0 28px}
        .lylo-card-person{text-align:center;min-width:0}
        .lylo-card-photo{position:relative;width:108px;height:108px;margin:0 auto 14px;border-radius:50%;overflow:hidden;border:1px solid rgba(255,255,255,.10);background:radial-gradient(circle at 35% 30%,rgba(98,157,226,.15),rgba(17,31,50,.82));box-shadow:0 16px 38px rgba(0,0,0,.22)}
        .lylo-card-photo img{width:100%;height:100%;display:block;object-fit:cover}
        .lylo-card-fallback{display:none;width:100%;height:100%;place-items:center;font-family:'Cormorant Garamond',serif;font-size:31px;color:#cdd9e7}
        .lylo-card-person-copy strong{display:block;margin-bottom:4px;color:#f1f5fa;font-family:'Cormorant Garamond',serif;font-size:24px;font-weight:400;line-height:1.05}
        .lylo-card-person-copy span{display:block;margin-bottom:4px;color:#9eacbd;font-size:10.5px;line-height:1.4}
        .lylo-card-person-copy small{display:block;color:#65778c;font-size:9px;line-height:1.4}
        .lylo-card-button{margin-top:auto;display:inline-flex;align-items:center;justify-content:center;gap:10px;min-height:47px;padding:0 18px;border-radius:999px;border:1px solid transparent;background:linear-gradient(180deg,rgba(16,29,48,.96),rgba(9,18,31,.98)) padding-box,linear-gradient(115deg,rgba(124,191,255,.64),rgba(111,221,183,.30),rgba(151,122,255,.44)) border-box;color:#f5f9ff!important;text-decoration:none;font-size:12.5px;font-weight:600;box-shadow:inset 0 1px 0 rgba(255,255,255,.08),0 12px 30px rgba(42,112,210,.10);transition:transform .2s ease,filter .2s ease}
        .lylo-card-button:hover{transform:translateY(-2px);filter:brightness(1.05)}
        .lylo-card-button span{font-size:16px}
        @media(max-width:979px){
          .lylo-community{padding:64px 20px!important}
          .lylo-community-inner{gap:12px!important}
          .lylo-community-card{min-height:0!important;padding:30px 22px!important;text-align:center!important;align-items:center!important}
          .lylo-community-card h3{font-size:35px!important;max-width:340px!important}
          .lylo-card-copy{max-width:340px;font-size:12.5px;line-height:1.62}
          .lylo-card-meta{width:100%;max-width:340px;margin:24px auto 0}
          .lylo-card-people{width:100%;max-width:340px;margin:7px auto 24px;gap:12px}
          .lylo-card-photo{width:98px;height:98px}
          .lylo-card-person-copy strong{font-size:22px}
          .lylo-card-person-copy span{font-size:9.5px}
          .lylo-card-person-copy small{font-size:8.5px}
          .lylo-card-button{width:100%;max-width:340px;margin:24px auto 0}
        }
        @media(max-width:390px){.lylo-card-photo{width:88px;height:88px}.lylo-card-people{gap:8px}}
      `;
      document.head.appendChild(style);
    }
  };

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', () => setTimeout(render, 0), {once:true});
  else setTimeout(render, 0);
})();