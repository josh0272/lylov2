(() => {
  const wave = document.getElementById("phoneWave");
  if (wave && !wave.children.length) {
    const heights = [
      6, 9, 12, 8, 15, 11, 19, 13, 24, 18, 30, 21, 36, 27, 44, 31, 52, 39, 60,
      45, 68, 52, 76, 58, 83, 64, 72, 55, 63, 48, 54, 41, 46, 35, 40, 31, 36,
      28, 42, 34, 49, 38, 58, 45, 66, 51, 73, 57, 81, 62, 88, 69, 78, 60, 70,
      54, 61, 47, 53, 41, 46, 36, 39, 31, 34, 27, 29, 23, 24, 19, 18, 14, 12, 9,
    ];
    heights.forEach((height, index) => {
      const bar = document.createElement("span");
      bar.className = `audio-bar${index === 0 ? " active" : ""}`;
      bar.style.height = `${height}%`;
      wave.appendChild(bar);
    });
  }

  const initStickyBooking = () => {
    const sticky = document.querySelector(".preview-mobile-booking");
    const hero = document.querySelector(".hero");
    const booking = document.getElementById("book-demo");
    if (!sticky || !hero || !booking) return;

    const mobile = window.matchMedia("(max-width: 620px)");
    const update = () => {
      if (!mobile.matches) {
        sticky.classList.remove("is-visible");
        document.body.classList.remove("preview-sticky-visible");
        return;
      }
      const heroPast = hero.getBoundingClientRect().bottom < 36;
      const bookingRect = booking.getBoundingClientRect();
      const bookingVisible =
        bookingRect.top < window.innerHeight * 0.84 && bookingRect.bottom > 100;
      const visible = heroPast && !bookingVisible;
      sticky.classList.toggle("is-visible", visible);
      document.body.classList.toggle("preview-sticky-visible", visible);
    };

    update();
    window.addEventListener("scroll", update, { passive: true });
    window.addEventListener("resize", update, { passive: true });
    mobile.addEventListener?.("change", update);
  };

  const initWorkflowSuggestion = () => {
    const section = document.querySelector(".lylo-other-workflows");
    const form = document.getElementById("et1-form-suggest");
    const input = document.getElementById("et1-form-input");
    const status = document.getElementById("et1-form-status");
    const submit = form?.querySelector('button[type="submit"]');
    const label = form?.querySelector("label");
    if (!section || !form || !input || !status || !submit || !label) return;

    const chips = Array.from(
      section.querySelectorAll(
        '.et1-form-set:not([aria-hidden="true"]) .et1-form-chip',
      ),
    );
    const selectValue = (value) => {
      if (form.classList.contains("is-submitted")) return;
      input.value = value;
      status.textContent = "";
      status.className = "et1-suggest-status";
      submit.textContent = "Send suggestion";
      submit.classList.add("is-ready");
      chips.forEach((chip) => {
        const selected = chip.textContent.trim() === value;
        chip.classList.toggle("is-selected", selected);
        chip.setAttribute("aria-pressed", selected ? "true" : "false");
      });
      form.scrollIntoView({ behavior: "smooth", block: "nearest" });
    };

    chips.forEach((chip) => {
      chip.setAttribute("aria-pressed", "false");
      chip.addEventListener("click", () =>
        selectValue(chip.textContent.trim()),
      );
      chip.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          selectValue(chip.textContent.trim());
        }
      });
    });

    input.addEventListener("input", () => {
      const value = input.value.trim();
      submit.textContent = value ? "Send suggestion" : "Suggest a form";
      submit.classList.toggle("is-ready", Boolean(value));
      if (status.textContent) {
        status.textContent = "";
        status.className = "et1-suggest-status";
      }
    });

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (form.classList.contains("is-submitted")) return;
      const value = input.value.trim();
      if (!value) {
        status.className = "et1-suggest-status error";
        status.textContent = "Add a form or workflow first.";
        input.focus();
        return;
      }

      submit.disabled = true;
      status.className = "et1-suggest-status";
      status.textContent = "Sending…";

      try {
        const body = new FormData();
        body.append("name", "Lylo preview workflow suggestion");
        body.append("email", "");
        body.append("answers", `Form or workflow suggestion: ${value}`);
        body.append("transcript", "");
        const response = await fetch("/api/submit", { method: "POST", body });
        const data = await response.json().catch(() => ({}));
        if (!response.ok || data.ok === false)
          throw new Error(data.error || "Could not send");

        form.classList.add("is-submitted");
        label.textContent = "Thanks — we’ve added it to the list.";
        status.className = "et1-suggest-status preview-suggest-success";
        status.innerHTML =
          '<span>If you’d like to see how Lylo could approach that workflow, <a href="/call#book">book a 20-minute demo</a>.</span>';
      } catch (_) {
        status.className = "et1-suggest-status error";
        status.textContent = "Could not send that just now. Please try again.";
        submit.disabled = false;
      }
    });
  };

  const polishChat = () => {
    const launcher = document.querySelector("#lylo-chat-launcher span");
    const panel = document.getElementById("lylo-chat-panel");
    if (!launcher || !panel || panel.dataset.previewPolished === "1")
      return false;
    panel.dataset.previewPolished = "1";
    launcher.textContent = "Ask about Lylo";
    panel.setAttribute(
      "aria-label",
      "Website assistant for questions about Lylo",
    );
    const heading = panel.querySelector(".lylo-chat-empty h2");
    const copy = panel.querySelector(".lylo-chat-empty p");
    const input = panel.querySelector("#lylo-chat-input");
    const note = panel.querySelector(".lylo-chat-note");
    if (heading) heading.textContent = "Ask about Lylo.";
    if (copy)
      copy.textContent =
        "Ask about the demonstrations, privacy approach or 20-minute call.";
    if (input) {
      input.placeholder = "Ask about Lylo";
      input.setAttribute("aria-label", "Ask the Lylo website assistant");
    }
    if (note)
      note.textContent =
        "Please don’t enter client information · AI responses are general information, not legal advice.";
    return true;
  };

  const initChatPolish = () => {
    if (polishChat()) return;
    const observer = new MutationObserver(() => {
      if (polishChat()) observer.disconnect();
    });
    observer.observe(document.body, { childList: true, subtree: true });
    window.setTimeout(() => observer.disconnect(), 10000);
  };

  const init = () => {
    initStickyBooking();
    initWorkflowSuggestion();
    initChatPolish();
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }
})();
