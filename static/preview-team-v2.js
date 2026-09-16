(() => {
  const joshPhoto = '/static/preview-joshua.jpg?v=4';
  const jessPhoto = '/static/preview-jessica.jpg?v=4';
  const aboutPath = '/static/preview-about.html';

  const render = () => {
    const research = document.querySelector('.lylo-research-card');
    const team = document.querySelector('.lylo-team-card');
    if (!research || !team) return;

    research.innerHTML = `
      <div class="lylo-v3-kicker">Help shape Lylo</div>
      <h3>We want to learn before we build more.</h3>
      <p class="lylo-v3-copy">Our research looks at how legal work is done today — including ET1s, Schedules of Loss, current workflows, common errors, AI concerns and the safeguards people would need before trusting a tool like Lylo.</p>
      <div class="lylo-v3-detail">
        <span>Research focus</span>
        <strong>Workflow · accuracy · AI concerns · security & control</strong>
      </div>
      <a class="lylo-v3-button" href="/research">Take the questionnaire <span>→</span></a>
    `;

    team.innerHTML = `
      <div class="lylo-v3-kicker">The people behind Lylo</div>
      <h3>Joshua Sam & Jessica Jayan</h3>
      <div class="lylo-v3-people">
        <div class="lylo-v3-person">
          <img src="${joshPhoto}" alt="Joshua Sam" loading="eager">
          <div><strong>Joshua Sam</strong><span>Product & technology</span><small>MEng Electrical & Mechanical Engineering · Strathclyde · 2:1</small></div>
        </div>
        <div class="lylo-v3-person">
          <img src="${jessPhoto}" alt="Jessica Jayan" loading="eager">
          <div><strong>Jessica Jayan</strong><span>Legal research & workflow input</span><small>Scots (Clinical) LLB · DPLP · Strathclyde Law Clinic</small></div>
        </div>
      </div>
      <p class="lylo-v3-team-copy">See our backgrounds, the work we are doing on Lylo and why we are using research and pilot testing before taking the product further.</p>
      <a class="lylo-v3-button" href="${aboutPath}">About us <span>→</span></a>
    `;

    let style = document.getElementById('preview-team-v3-styles');
    if (!style) {
      style = document.createElement('style');
      style.id = 'preview-team-v3-styles';
      style.textContent = `
        .lylo-community-inner{align-items:stretch}
        .lylo-community-card{min-height:430px!important;padding:38px 36px!important}
        .lylo-v3-kicker{margin-bottom:18px;color:#88a8cf;font-size:10px;font-weight:700;letter-spacing:.14em;text-transform:uppercase}
        .lylo-community-card h3{max-width:440px!important;margin:0 0 17px!important;font-size:40px!important;line-height:1.04!important}
        .lylo-v3-copy,.lylo-v3-team-copy{max-width:455px!important;margin:0!important;color:#96a5b8!important;font-size:13.5px!important;line-height:1.68!important}
        .lylo-v3-detail{margin:27px 0 0;padding:16px 0;border-top:1px solid rgba(255,255,255,.065);border-bottom:1px solid rgba(255,255,255,.065)}
        .lylo-v3-detail span{display:block;margin-bottom:6px;color:#66798f;font-size:9px;font-weight:700;letter-spacing:.12em;text-transform:uppercase}
        .lylo-v3-detail strong{display:block;color:#cbd6e3;font-size:11.5px;font-weight:500;line-height:1.55}
        .lylo-v3-people{width:100%;display:grid;grid-template-columns:1fr 1fr;gap:25px;margin:4px 0 22px}
        .lylo-v3-person{display:flex;align-items:center;gap:13px;min-width:0}
        .lylo-v3-person img{width:78px;height:78px;flex:0 0 78px;border-radius:50%;object-fit:cover;border:1px solid rgba(255,255,255,.10);box-shadow:0 12px 28px rgba(0,0,0,.20)}
        .lylo-v3-person strong{display:block;margin-bottom:4px;color:#f1f5fa;font-family:'Cormorant Garamond',serif;font-size:23px;font-weight:400;line-height:1}
        .lylo-v3-person span{display:block;margin-bottom:4px;color:#a2b0c1;font-size:10.5px;line-height:1.35}
        .lylo-v3-person small{display:block;max-width:190px;color:#66798f;font-size:9px;line-height:1.4}
        .lylo-v3-button{margin-top:auto;display:inline-flex;align-items:center;justify-content:center;gap:10px;min-height:48px;padding:0 19px;border-radius:999px;border:1px solid transparent;background:linear-gradient(180deg,rgba(16,29,48,.96),rgba(9,18,31,.98)) padding-box,linear-gradient(115deg,rgba(124,191,255,.72),rgba(111,221,183,.38),rgba(151,122,255,.52)) border-box;color:#f5f9ff!important;text-decoration:none;font-size:13px;font-weight:600;box-shadow:inset 0 1px 0 rgba(255,255,255,.09),0 12px 34px rgba(42,112,210,.13);transition:transform .2s ease,filter .2s ease}
        .lylo-v3-button:hover{transform:translateY(-2px);filter:brightness(1.06)}
        .lylo-v3-button span{font-size:16px}
        @media(max-width:979px){
          .lylo-community-card{min-height:0!important;padding:27px 21px!important}
          .lylo-community-card h3{font-size:34px!important;max-width:340px!important}
          .lylo-v3-copy,.lylo-v3-team-copy{max-width:340px!important;font-size:12.5px!important;line-height:1.62!important}
          .lylo-v3-detail{width:100%;max-width:340px;margin:22px auto 0}
          .lylo-v3-people{max-width:340px;margin:4px auto 20px;grid-template-columns:1fr;gap:16px;text-align:left}
          .lylo-v3-person{gap:14px}
          .lylo-v3-person img{width:72px;height:72px;flex-basis:72px}
          .lylo-v3-person strong{font-size:22px}
          .lylo-v3-person small{max-width:225px}
          .lylo-v3-button{width:100%;max-width:340px;margin-left:auto;margin-right:auto;margin-top:21px}
        }
      `;
      document.head.appendChild(style);
    }
  };

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', () => setTimeout(render, 0), {once:true});
  else setTimeout(render, 0);
})();