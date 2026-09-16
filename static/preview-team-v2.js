(() => {
  const aboutPath = '/static/preview-about.html';

  const render = () => {
    const research = document.querySelector('.lylo-research-card');
    const team = document.querySelector('.lylo-team-card');
    if (!research || !team) return;

    research.innerHTML = `
      <div class="lylo-v4-kicker">Help shape Lylo</div>
      <h3>Help us make Lylo better.</h3>
      <p class="lylo-v4-copy">If you work or study in law, tell us what takes time, what you would trust AI to help with, and what would make a tool like Lylo useful to you.</p>
      <a class="lylo-v4-button" href="/research">Take the questionnaire <span>→</span></a>
    `;

    team.innerHTML = `
      <div class="lylo-v4-kicker">The people behind Lylo</div>
      <h3>Joshua Sam & Jessica Jayan</h3>
      <div class="lylo-v4-people">
        <div class="lylo-v4-person"><strong>Joshua Sam</strong><span>Product & technology</span><small>MEng Electrical & Mechanical Engineering · Strathclyde · 2:1</small></div>
        <div class="lylo-v4-person"><strong>Jessica Jayan</strong><span>Legal research & workflow input</span><small>Scots (Clinical) LLB · DPLP · Strathclyde Law Clinic</small></div>
      </div>
      <a class="lylo-v4-button" href="${aboutPath}">About us <span>→</span></a>
    `;

    let style = document.getElementById('preview-team-v4-styles');
    if (!style) {
      style = document.createElement('style');
      style.id = 'preview-team-v4-styles';
      style.textContent = `
        .lylo-community-inner{align-items:stretch;gap:18px}
        .lylo-community-card{min-height:360px!important;padding:42px 40px!important}
        .lylo-v4-kicker{margin-bottom:22px;color:#88a8cf;font-size:10px;font-weight:700;letter-spacing:.14em;text-transform:uppercase}
        .lylo-community-card h3{max-width:440px!important;margin:0 0 18px!important;font-size:39px!important;line-height:1.04!important}
        .lylo-v4-copy{max-width:440px!important;margin:0!important;color:#96a5b8!important;font-size:14px!important;line-height:1.72!important}
        .lylo-v4-people{display:grid;gap:0;margin:3px 0 22px;border-top:1px solid rgba(255,255,255,.065);border-bottom:1px solid rgba(255,255,255,.065)}
        .lylo-v4-person{padding:17px 0}
        .lylo-v4-person+.lylo-v4-person{border-top:1px solid rgba(255,255,255,.055)}
        .lylo-v4-person strong{display:block;margin-bottom:5px;color:#f1f5fa;font-family:'Cormorant Garamond',serif;font-size:24px;font-weight:400;line-height:1}
        .lylo-v4-person span{display:block;margin-bottom:4px;color:#a2b0c1;font-size:11px;line-height:1.35}
        .lylo-v4-person small{display:block;color:#66798f;font-size:9.5px;line-height:1.45}
        .lylo-v4-button{margin-top:auto;display:inline-flex;align-items:center;justify-content:center;gap:10px;min-height:47px;padding:0 18px;border-radius:999px;border:1px solid transparent;background:linear-gradient(180deg,rgba(16,29,48,.96),rgba(9,18,31,.98)) padding-box,linear-gradient(115deg,rgba(124,191,255,.72),rgba(111,221,183,.38),rgba(151,122,255,.52)) border-box;color:#f5f9ff!important;text-decoration:none;font-size:12.5px;font-weight:600;box-shadow:inset 0 1px 0 rgba(255,255,255,.08),0 12px 30px rgba(42,112,210,.11);transition:transform .2s ease,filter .2s ease}
        .lylo-v4-button:hover{transform:translateY(-2px);filter:brightness(1.05)}
        .lylo-v4-button span{font-size:16px}
        @media(max-width:979px){
          .lylo-community-card{min-height:0!important;padding:29px 22px!important}
          .lylo-community-card h3{font-size:34px!important;max-width:340px!important}
          .lylo-v4-copy{max-width:340px!important;font-size:12.7px!important;line-height:1.65!important}
          .lylo-v4-people{width:100%;max-width:340px;margin:2px auto 20px;text-align:left}
          .lylo-v4-button{width:100%;max-width:340px;margin:22px auto 0}
        }
      `;
      document.head.appendChild(style);
    }
  };

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', () => setTimeout(render, 0), {once:true});
  else setTimeout(render, 0);
})();