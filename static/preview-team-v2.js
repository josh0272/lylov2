(() => {
  const joshPhoto = '/static/preview-joshua.jpg?v=3';
  const jessPhoto = '/static/preview-jessica.jpg?v=3';
  const aboutPath = '/static/preview-about.html';

  const render = () => {
    const research = document.querySelector('.lylo-research-card');
    const team = document.querySelector('.lylo-team-card');
    if (!research || !team) return;

    research.innerHTML = `
      <div class="lylo-v2-kicker">Help shape Lylo</div>
      <h3>Work or study in law?</h3>
      <p class="lylo-v2-lead">We want to learn which parts of legal work waste the most time, what people would actually trust AI to help with, and what Lylo should focus on next.</p>
      <div class="lylo-research-audience">Solicitors · trainees · law students</div>
      <a class="lylo-v2-button" href="/research">Take the questionnaire <span>→</span></a>
      <div class="lylo-v2-note">Short questionnaire · no sales pitch.</div>
    `;

    team.innerHTML = `
      <div class="lylo-v2-kicker">The people behind Lylo</div>
      <div class="lylo-linkedin-pair">
        <div class="lylo-linkedin-person">
          <img src="${joshPhoto}" alt="Joshua Sam" loading="eager">
          <strong>Joshua Sam</strong>
          <span>Engineering · product & technology</span>
          <small>MEng · University of Strathclyde</small>
        </div>
        <div class="lylo-linkedin-person">
          <img src="${jessPhoto}" alt="Jessica Jayan" loading="eager">
          <strong>Jessica Jayan</strong>
          <span>Law · DPLP</span>
          <small>Strathclyde Law Clinic</small>
        </div>
      </div>
      <p class="lylo-v2-team-copy">Two different backgrounds are helping us test both sides of Lylo: whether the technology works and whether it makes sense for legal work.</p>
      <a class="lylo-v2-button" href="${aboutPath}">View our profiles <span>→</span></a>
    `;

    let style = document.getElementById('preview-team-v2-styles');
    if (!style) {
      style = document.createElement('style');
      style.id = 'preview-team-v2-styles';
      style.textContent = `
        .lylo-community-inner{align-items:stretch}
        .lylo-community-card{min-height:500px!important;padding:38px 36px!important}
        .lylo-v2-kicker{color:#88a8cf;font-size:10px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;margin-bottom:19px}
        .lylo-community-card h3{font-size:43px!important;line-height:1.02!important;max-width:430px!important;margin:0 0 17px!important}
        .lylo-v2-lead,.lylo-v2-team-copy{color:#96a5b8!important;font-size:14px!important;line-height:1.68!important;margin:0!important;max-width:445px!important}
        .lylo-research-audience{margin:30px 0 0;padding:14px 0;border-top:1px solid rgba(255,255,255,.07);border-bottom:1px solid rgba(255,255,255,.07);color:#cbd5e1;font-size:12px;letter-spacing:.015em}
        .lylo-linkedin-pair{width:100%;display:grid;grid-template-columns:1fr 1fr;gap:26px;margin:8px 0 25px}
        .lylo-linkedin-person{text-align:center;min-width:0}
        .lylo-linkedin-person img{display:block;width:158px;height:158px;margin:0 auto 15px;border-radius:50%;object-fit:cover;border:2px solid rgba(255,255,255,.10);box-shadow:0 18px 44px rgba(0,0,0,.25)}
        .lylo-linkedin-person strong{display:block;color:#f3f7fb;font-family:'Cormorant Garamond',serif;font-size:27px;font-weight:400;line-height:1.05;margin-bottom:6px}
        .lylo-linkedin-person span{display:block;color:#a1afc0;font-size:11px;line-height:1.4;margin-bottom:3px}
        .lylo-linkedin-person small{display:block;color:#68798e;font-size:9.5px;line-height:1.4}
        .lylo-v2-button{margin-top:auto;display:inline-flex;align-items:center;justify-content:center;gap:10px;min-height:48px;padding:0 19px;border-radius:999px;border:1px solid transparent;background:linear-gradient(180deg,rgba(16,29,48,.96),rgba(9,18,31,.98)) padding-box,linear-gradient(115deg,rgba(124,191,255,.72),rgba(111,221,183,.38),rgba(151,122,255,.52)) border-box;color:#f5f9ff!important;text-decoration:none;font-size:13px;font-weight:600;box-shadow:inset 0 1px 0 rgba(255,255,255,.09),0 12px 34px rgba(42,112,210,.13);transition:transform .2s ease,filter .2s ease}
        .lylo-v2-button:hover{transform:translateY(-2px);filter:brightness(1.06)}
        .lylo-v2-button span{font-size:16px}
        .lylo-v2-note{margin-top:9px;color:#63758b;font-size:10px;line-height:1.45}
        .lylo-team-card .lylo-v2-button{margin-top:25px}
        @media(max-width:979px){
          .lylo-community-card{min-height:0!important;padding:28px 21px!important}
          .lylo-community-card h3{font-size:35px!important;max-width:335px!important}
          .lylo-v2-lead,.lylo-v2-team-copy{font-size:12.5px!important;max-width:340px!important}
          .lylo-research-audience{width:100%;max-width:335px;margin:23px auto 0}
          .lylo-linkedin-pair{max-width:350px;gap:12px;margin:7px auto 21px}
          .lylo-linkedin-person img{width:132px;height:132px;margin-bottom:12px}
          .lylo-linkedin-person strong{font-size:22px}
          .lylo-linkedin-person span{font-size:9.5px}
          .lylo-linkedin-person small{font-size:8.5px}
          .lylo-v2-button{width:100%;max-width:335px;margin-left:auto;margin-right:auto}
          .lylo-team-card .lylo-v2-button{margin-top:20px}
        }
        @media(max-width:390px){.lylo-linkedin-person img{width:118px;height:118px}}
      `;
      document.head.appendChild(style);
    }
  };

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', () => setTimeout(render, 0), {once:true});
  else setTimeout(render, 0);
})();