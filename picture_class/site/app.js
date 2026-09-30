(() => {
  const slides = [...document.querySelectorAll(".slide")];
  const nav = document.querySelector(".spine-nav");
  const only = new URLSearchParams(location.search).get("only");
  if (only) slides.forEach((s) => { if (s.id !== only) s.style.display = "none"; });

  slides.forEach((s, i) => {
    const a = document.createElement("a");
    a.href = `#${s.id}`;
    a.dataset.region = s.dataset.region;
    a.dataset.title = `${i + 1}. ${s.dataset.title}`;
    a.setAttribute("aria-label", a.dataset.title);
    nav.appendChild(a);
  });
  const dots = [...nav.children];

  let current = 0;
  const setCurrent = (i) => {
    current = i;
    dots.forEach((d, k) => d.classList.toggle("is-current", k === i));
  };

  const io = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting) setCurrent(slides.indexOf(e.target));
    });
  }, { rootMargin: "-45% 0px -50% 0px" });
  slides.forEach((s) => io.observe(s));

  const go = (i) => {
    i = Math.max(0, Math.min(slides.length - 1, i));
    slides[i].scrollIntoView({ block: "start" });
  };

  document.addEventListener("keydown", (e) => {
    if (e.target.closest("input, textarea, summary")) return;
    const k = e.key;
    if (["ArrowDown", "ArrowRight", "PageDown", " "].includes(k)) { e.preventDefault(); go(current + 1); }
    else if (["ArrowUp", "ArrowLeft", "PageUp"].includes(k)) { e.preventDefault(); go(current - 1); }
    else if (k === "Home") { e.preventDefault(); go(0); }
    else if (k === "End") { e.preventDefault(); go(slides.length - 1); }
    else if (k === "n" || k === "N") document.body.classList.toggle("show-notes");
    else if (k === "f" || k === "F") {
      if (!document.fullscreenElement) document.documentElement.requestFullscreen?.();
      else document.exitFullscreen?.();
    }
  });

  // ---------------------------------------------------------- slice scrubber
  const canvas = document.getElementById("scrub-canvas");
  if (!canvas) return;
  const range = document.getElementById("scrub-range");
  const labelToggle = document.getElementById("scrub-labels");
  const readout = document.getElementById("scrub-readout");
  const ctx = canvas.getContext("2d");
  const load = (src) => new Promise((res) => { const im = new Image(); im.onload = () => res(im); im.src = src; });

  Promise.all([fetch("assets/sprite.json").then((r) => r.json()), load("assets/sprite_ct.jpg"), load("assets/sprite_lab.jpg")])
    .then(([meta, ct, lab]) => {
      canvas.width = meta.w;
      canvas.height = meta.h;
      range.max = meta.n - 1;
      range.value = Math.floor(meta.n / 2);
      const draw = () => {
        const i = +range.value;
        const sx = (i % meta.cols) * meta.w;
        const sy = Math.floor(i / meta.cols) * meta.h;
        ctx.drawImage(labelToggle.checked ? lab : ct, sx, sy, meta.w, meta.h, 0, 0, meta.w, meta.h);
        const mm = (i - (meta.n - 1) / 2) * meta.mm_per_slice;
        readout.textContent = `第 ${i + 1} / ${meta.n} 层，相对视野中心偏${mm >= 0 ? "右" : "左"} ${Math.abs(mm).toFixed(0)} mm`;
      };
      range.addEventListener("input", draw);
      labelToggle.addEventListener("change", draw);
      canvas.addEventListener("wheel", (e) => {
        e.preventDefault();
        range.value = +range.value + (e.deltaY > 0 ? 1 : -1);
        draw();
      }, { passive: false });
      draw();
    });
})();
