(() => {
  const joshPhoto = '/static/preview-joshua.jpg';
  const jessPhoto = '/static/preview-jessica.jpg';
  const aboutPath = '/static/preview-about.html';

  const render = () => {
    const research = document.querySelector('.lylo-research-card');
    const team = document.querySelector('.lylo-team-card');
    if (!research || !team) return;

    research.innerHTML = `
      <div class="lylo-v2-kicker">Help shape Lylo</div>
      <h3>Help shape what Lylo becomes.</h3>
      <p class="lylo-v2-lead">If you work or study in law, tell us where time is being lost and where AI should — or should not — help.</p>
      <div class="lylo-question-list" aria-label="Example research questions">
        <div><span>01</span><strong>What takes too long?</strong></div>
        <div><span>02</span><strong>Where could AI help?</strong></div>
        <div><span>03</span><strong>Where should it stay out?</strong></div>
      </div>
      <a class="lylo-v2-button" href="/research">Take the questionnaire <span>→</span></a>
      <div class="lylo-v2-note">For solicitors, trainees and law students.</div>
    `;

    team.innerHTML = `
      <div class="lylo-v2-kicker">The people behind Lylo</div>
      <div class="lylo-founder-pair">
        <figure>
          <img src="${joshPhoto}" alt="Joshua Sam" loading="eager">
          <figcaption><strong>Joshua Sam</strong><span>Engineering · product & technology</span></figcaption>
        </figure>
        <figure>
          <img src="${jessPhoto}" alt="Jessica Jayan" loading="eager">
          <figcaption><strong>Jessica Jayan</strong><span>Law · DPLP · Strathclyde Law Clinic</span></figcaption>
        </figure>
      </div>
      <p class="lylo-v2-team-copy">Engineering builds and tests the product. Legal experience helps us judge whether it actually fits the way firms work.</p>
      <a class="lylo-v2-button" href="${aboutPath}">Meet Joshua & Jessica <span>→</span></a>
    `;

    let style = document.getElementById('preview-team-v2-styles');
    if (!style) {
      style = document.createElement('style');
      style.id = 'preview-team-v2-styles';
      style.textContent = `
        .lylo-community-inner{align-items:stretch}
        .lylo-community-card{min-height:520px!important;padding:38px 36px!important}
        .lylo-v2-kicker{color:#88a8cf;font-size:10px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;margin-bottom:19px}
        .lylo-community-card h3{font-size:42px!important;line-height:1.02!important;max-width:430px!important;margin:0 0 15px!important}
        .lylo-v2-lead,.lylo-v2-team-copy{color:#96a5b8!important;font-size:13.5px!important;line-height:1.65!important;margin:0!important;max-width:445px!important}
        .lylo-question-list{width:100%;margin:29px 0 28px;border-top:1px solid rgba(255,255,255,.07)}
        .lylo-question-list>div{display:grid;grid-template-columns:34px 1fr;gap:13px;align-items:center;padding:15px 0;border-bottom:1px solid rgba(255,255,255,.07)}
        .lylo-question-list span{color:#63758b;font-size:10px;font-weight:700;letter-spacing:.1em}
        .lylo-question-list strong{color:#e7eef7;font-size:13px;font-weight:500}
        .lylo-founder-pair{width:100%;display:grid;grid-template-columns:1fr 1fr;gap:18px;margin:3px 0 23px}
        .lylo-founder-pair figure{margin:0;text-align:center}
        .lylo-founder-pair img{display:block;width:min(100%,190px);aspect-ratio:1/1;margin:0 auto 13px;object-fit:cover;border-radius:50%;border:1px solid rgba(255,255,255,.10);box-shadow:0 18px 45px rgba(0,0,0,.23)}
        .lylo-founder-pair strong{display:block;color:#f3f7fb;font-family:'Cormorant Garamond',serif;font-size:25px;font-weight:400;line-height:1.05;margin-bottom:5px}
        .lylo-founder-pair span{display:block;max-width:185px;margin:auto;color:#7f91a7;font-size:10px;line-height:1.4}
        .lylo-v2-button{margin-top:auto;display:inline-flex;align-items:center;justify-content:center;gap:10px;min-height:48px;padding:0 19px;border-radius:999px;border:1px solid transparent;background:linear-gradient(180deg,rgba(16,29,48,.96),rgba(9,18,31,.98)) padding-box,linear-gradient(115deg,rgba(124,191,255,.72),rgba(111,221,183,.38),rgba(151,122,255,.52)) border-box;color:#f5f9ff!important;text-decoration:none;font-size:13px;font-weight:600;box-shadow:inset 0 1px 0 rgba(255,255,255,.09),0 12px 34px rgba(42,112,210,.13);transition:transform .2s ease,filter .2s ease}
        .lylo-v2-button:hover{transform:translateY(-2px);filter:brightness(1.06)}
        .lylo-v2-button span{font-size:16px}
        .lylo-v2-note{margin-top:9px;color:#63758b;font-size:10px;line-height:1.45}
        .lylo-team-card .lylo-v2-button{margin-top:25px}
        @media(max-width:979px){
          .lylo-community-card{min-height:0!important;padding:27px 21px!important}
          .lylo-community-card h3{font-size:35px!important;max-width:335px!important}
          .lylo-v2-lead,.lylo-v2-team-copy{font-size:12.5px!important;max-width:340px!important}
          .lylo-question-list{max-width:335px;margin:22px auto 23px;text-align:left}
          .lylo-founder-pair{max-width:350px;gap:14px;margin:4px auto 20px}
          .lylo-founder-pair img{width:min(100%,145px);margin-bottom:11px}
          .lylo-founder-pair strong{font-size:22px}
          .lylo-founder-pair span{font-size:9px;max-width:145px}
          .lylo-v2-button{width:100%;max-width:335px;margin-left:auto;margin-right:auto}
          .lylo-team-card .lylo-v2-button{margin-top:20px}
        }
        @media(max-width:430px){
          .lylo-founder-pair img{width:132px}
          .lylo-founder-pair{gap:10px}
        }
      `;
      document.head.appendChild(style);
    }
  };

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', () => setTimeout(render, 0), {once:true});
  else setTimeout(render, 0);
})();