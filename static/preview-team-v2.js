(() => {
  const removeCommunity = () => {
    document.querySelector('.lylo-community')?.remove();
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => setTimeout(removeCommunity, 0), {once:true});
  } else {
    setTimeout(removeCommunity, 0);
  }
})();