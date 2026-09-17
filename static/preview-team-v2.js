(() => {
  const aboutPath = '/static/preview-about.html';

  const person = (image, initials, name, role) => `
    <div class="lylo-sub-person">
      <div class="lylo-sub-photo">
        <img src="${image}" alt="${name}" onerror="this.style.display='none';this.nextElementSibling.style.display='grid'">
        <span class="lylo-sub-fallback">${initials}</span>
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

    community.classList.add('lylo-subsections');
    research.classList.add('lylo-subsection', 'lylo-research-subsection');
    team.classList.add('lylo-subsection', 'lylo-team-subsection');

    research.innerHTML = `
      <div class="lylo-subsection-inner">
        <h3>Help us make Lylo better.</h3>
        <p>If you work or study in law, your answers will help us decide what to improve and what to build next.</p>
        <a class="demo-cta lylo-sub-cta" href="/research">Take the questionnaire</a>
      </div>
    `;

    team.innerHTML = `
      <div class="lylo-subsection-inner">
        <h3>Meet the people behind Lylo.</h3>
        <div class="lylo-sub-people">
          ${person('/static/joshua-profile.jpg', 'JS', 'Joshua Sam', 'Product & technology')}
          ${person('/static/jessica-profile.jpg', 'JJ', 'Jessica Jayan', 'Legal research & workflow')}
        </div>
        <a class="demo-cta lylo-sub-cta" href="${aboutPath}">Meet us</a>
      </div>
    `;

    let style = document.getElementById('preview-subsection-styles');
    if (!style) {
      style = document.createElement('style');
      style.id = 'preview-subsection-styles';
      style.textContent = `
        .lylo-subsections{padding:0 var(--gutter)!important;border-top:1px solid rgba(255,255,255,.055)!important;background:none!important}
        .lylo-subsections .lylo-community-inner{display:block!important;width:100%!important;max-width:980px!important;margin:0 auto!important;padding:0!important}
        .lylo-subsections .lylo-community-card{display:block!important;width:100%!important;max-width:none!important;min-height:0!important;margin:0!important;padding:0!important;border:0!important;border-radius:0!important;background:none!important;box-shadow:none!important;overflow:visible!important;text-align:center!important}
        .lylo-subsections .lylo-community-card::after{display:none!important}
        .lylo-subsection{padding:68px 0 64px!important}
        .lylo-subsection+.lylo-subsection{border-top:1px solid rgba(255,255,255,.065)!important}
        .lylo-subsection-inner{width:100%;max-width:980px;margin:0 auto;text-align:center}
        .lylo-subsection h3{margin:0 0 8px!important;font-family:'Cormorant Garamond',serif!important;font-size:clamp(28px,3vw,36px)!important;font-weight:400!important;line-height:1.08!important;letter-spacing:-.02em!important;color:#eef4fb!important}
        .lylo-subsection p{max-width:520px!important;margin:0 auto!important;color:#8f9daf!important;font-size:14px!important;line-height:1.55!important}
        .lylo-sub-cta{margin-top:24px!important;min-height:44px!important;padding:0 18px!important;font-size:12.5px!important}
        .lylo-sub-people{display:flex;justify-content:center;gap:58px;margin:27px auto 0}
        .lylo-sub-person{width:150px;text-align:center}
        .lylo-sub-photo{position:relative;width:84px;height:84px;margin:0 auto 11px;border-radius:50%;overflow:hidden;border:1px solid rgba(255,255,255,.10);background:#101a29;box-shadow:0 12px 28px rgba(0,0,0,.18)}
        .lylo-sub-photo img{display:block;width:100%;height:100%;object-fit:cover}
        .lylo-sub-fallback{display:none;width:100%;height:100%;place-items:center;font-family:'Cormorant Garamond',serif;font-size:28px;color:#cdd9e7;background:radial-gradient(circle at 35% 30%,rgba(98,157,226,.13),rgba(17,31,50,.88))}
        .lylo-sub-person strong{display:block;margin-bottom:4px;color:#eef4fb;font-size:13px;font-weight:600;line-height:1.3}
        .lylo-sub-person>span{display:block;color:#7f91a7;font-size:10px;line-height:1.4}
        .lylo-team-subsection .lylo-sub-cta{margin-top:25px!important}
        @media(max-width:979px){
          .lylo-subsections{padding:0 20px!important}
          .lylo-subsection{padding:52px 0 48px!important}
          .lylo-subsection h3{font-size:27px!important;line-height:1.08!important;margin-bottom:7px!important}
          .lylo-subsection p{max-width:350px!important;font-size:13px!important;line-height:1.48!important}
          .lylo-sub-cta{width:auto!important;min-width:0!important;margin-top:20px!important}
          .lylo-sub-people{gap:30px;margin-top:23px}
          .lylo-sub-person{width:128px}
          .lylo-sub-photo{width:76px;height:76px;margin-bottom:9px}
          .lylo-sub-person strong{font-size:12px}
          .lylo-sub-person>span{font-size:9px}
        }
      `;
      document.head.appendChild(style);
    }
  };

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', () => setTimeout(render, 0), {once:true});
  else setTimeout(render, 0);
})();