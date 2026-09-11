(function () {
  function headingText(heading) {
    return Array.from(heading.childNodes)
      .filter(function (node) {
        return !(node.classList && node.classList.contains("headerlink"));
      })
      .map(function (node) {
        return node.textContent;
      })
      .join(" ")
      .replace(/\s+/g, " ")
      .trim();
  }

  function setupSectionMenu() {
    var article = document.querySelector(".md-content__inner");

    if (!article || article.querySelector(".section-jump")) {
      return;
    }

    var headings = Array.from(article.querySelectorAll("h1[id], h2[id], h3[id]"))
      .filter(function (heading) {
        return heading.id && headingText(heading);
      });

    if (headings.length < 3) {
      return;
    }

    var menu = document.createElement("nav");
    menu.className = "section-jump";
    menu.setAttribute("aria-label", "Navegacao por secoes");

    var label = document.createElement("span");
    label.className = "section-jump__label";
    label.textContent = "Seções";

    var control = document.createElement("div");
    control.className = "section-jump__control";

    var select = document.createElement("select");
    select.className = "section-jump__select";

    headings.forEach(function (heading) {
      var option = document.createElement("option");
      var depth = Number(heading.tagName.slice(1));
      option.value = "#" + heading.id;
      option.textContent = (depth > 1 ? "  ".repeat(depth - 2) : "") + headingText(heading);
      select.appendChild(option);
    });

    select.addEventListener("change", function () {
      var target = document.querySelector(select.value);

      if (target) {
        target.scrollIntoView({ behavior: "smooth", block: "start" });
        history.replaceState(null, "", select.value);
      }
    });

    function syncActiveHeading() {
      var active = headings[0];
      var offset = 130;

      headings.forEach(function (heading) {
        if (heading.getBoundingClientRect().top <= offset) {
          active = heading;
        }
      });

      if (active) {
        select.value = "#" + active.id;
      }
    }

    control.appendChild(select);
    menu.appendChild(label);
    menu.appendChild(control);
    article.insertBefore(menu, article.firstElementChild ? article.firstElementChild.nextSibling : null);

    syncActiveHeading();
    document.addEventListener("scroll", syncActiveHeading, { passive: true });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", setupSectionMenu);
  } else {
    setupSectionMenu();
  }

  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(setupSectionMenu);
  }
})();
