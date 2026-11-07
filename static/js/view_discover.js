(function () {
    "use strict";
    const { h } = PL;
    let resultsBox, input, searchBtn, lastResults = [];

    // ---------- search ----------
    async function search(q) {
        q = (q || "").trim();
        if (!q) return PL.toast("Type a topic first, for example “graph neural networks”.", "error");
        searchBtn.disabled = true;
        PL.loading(resultsBox, "Finding papers about “" + q + "”…");
        try {
            const d = await PL.api("/api/search?limit=12&q=" + encodeURIComponent(q));
            lastResults = d.results;
            if (!d.results.length) return void resultsBox.replaceChildren(h("div", { class: "empty" }, "No papers found. Try different or fewer words."));
            resultsBox.replaceChildren(h("h2", { class: "section__title", style: "margin:.2rem 0 .8rem" }, `${d.count} papers about “${q}”`), h("div", { class: "grid grid--2" }, d.results.map(card)));
        } catch (e) { PL.error(resultsBox, e, () => search(q)); } finally { searchBtn.disabled = false; }
    }

    function card(p) {
        const abs = h("p", { class: "paper__sum", style: "-webkit-line-clamp:4" }, p.abstract || "No abstract available.");
        const meta = [PL.fmt.authors(p.authors), p.year, p.venue].filter(Boolean).join(" · ");
        const addBtn = h("button", { class: "btn btn--small btn--ghost", type: "button" }, "Add to my library");
        const actions = h("div", { class: "row", style: "margin-top:.5rem" },
            h("button", { class: "btn btn--small btn--primary", type: "button", onclick: () => analyze({ title: p.title, text: p.abstract, paper: p }) }, "Analyze"),
            p.in_library ? h("button", { class: "btn btn--small btn--accent", type: "button", onclick: () => PL.ask("Explain this paper in simple terms.", { id: p.in_library, title: p.title }) }, "Ask the assistant")
                : addBtn,
            p.url ? h("a", { class: "btn btn--small btn--ghost", href: p.url, target: "_blank", rel: "noopener" }, "Open paper") : null);
        if (!p.in_library) addBtn.addEventListener("click", () => {
            if (addBtn.dataset.paperId) PL.ask("Explain this paper in simple terms.", { id: addBtn.dataset.paperId, title: p.title });
            else addToLibrary(p, addBtn);
        });
        return h("article", { class: "card paper", style: "cursor:default" }, h("div", { class: "paper__title" }, p.title), h("div", { class: "paper__meta" }, meta), abs,
            (p.abstract || "").length > 260 ? h("button", { class: "chart__toggle", type: "button", onclick: (e) => { abs.style.webkitLineClamp = abs.style.webkitLineClamp === "unset" ? "4" : "unset"; e.target.textContent = abs.style.webkitLineClamp === "unset" ? "Show less" : "Read more"; } }, "Read more") : null,
            actions);
    }

    // ---------- add to library (read the full paper so the assistant can discuss it) ----------
    async function addToLibrary(p, btn) {
        btn.disabled = true; btn.textContent = "Adding…";
        try {
            const r = await PL.api("/api/library/add", { json: { title: p.title, abstract: p.abstract, authors: p.authors, url: p.url, pdf_url: p.pdf_url, source: p.source, year: p.year, venue: p.venue, doi: p.doi, citation_count: p.citation_count } });
            if (r.status === "ready") return ready(btn, r.paper_id);
            btn.textContent = "Reading the paper… (about a minute)";
            for (let i = 0; i < 80; i++) {
                await new Promise((res) => setTimeout(res, 3000));
                const s = await PL.api("/api/library/add/status?id=" + encodeURIComponent(r.paper_id));
                if (s.status === "ready") return ready(btn, r.paper_id);
                if (s.status === "failed") throw new Error(s.message);
            }
            throw new Error("This is taking longer than expected. Please check your library in a minute.");
        } catch (e) { btn.disabled = false; btn.textContent = "Add to my library"; PL.toast(e.message, "error"); }
    }
    function ready(btn, id) {
        btn.dataset.paperId = id;                                  // the click handler switches from "add" to "ask" once the id is known
        btn.disabled = false; btn.className = "btn btn--small btn--accent"; btn.textContent = "Ready: ask the assistant";
        PL.toast("Added. You can now ask the assistant about this paper.", "success");
    }

    // ---------- analysis ----------
    async function analyze({ title, text, file, paper }) {
        const box = h("div", {}, h("button", { class: "btn btn--small btn--ghost drawer__close", type: "button", onclick: PL.closeDrawer }, "✕ Close"),
            title ? h("h2", { style: "font-size:1.2rem;margin:.3rem 0 .6rem" }, title) : h("h2", { style: "font-size:1.2rem;margin:.3rem 0 .6rem" }, "Your paper"));
        const body = h("div", {}); box.append(body); PL.openDrawer(box);
        PL.loading(body, "Reading and summarizing… this can take a little while the first time.");
        try {
            let d;
            if (file) { const f = new FormData(); f.append("file", file); d = await PL.api("/api/analyze", { method: "POST", body: f }); }
            else d = await PL.api("/api/analyze", { json: { text, title } });
            const simText = (d.summary || text || "").slice(0, 1000);
            body.replaceChildren(
                h("section", {}, h("h3", { class: "small muted" }, "SUMMARY"), h("p", {}, d.summary || "No summary available.")),
                d.keywords.length ? h("section", {}, h("h3", { class: "small muted" }, "KEYWORDS"), h("div", { class: "tagrow" }, d.keywords.map((k) => h("span", { class: "chip" }, k.word)))) : null,
                d.topics.length ? h("section", { style: "margin-top:.8rem" }, h("h3", { class: "small muted" }, "RELATED TOPICS"), h("div", { class: "tagrow" }, d.topics.slice(0, 4).map((t) => h("span", { class: "chip chip--accent" }, t.label)))) : null,
                h("section", { style: "margin-top:.8rem" }, h("h3", { class: "small muted" }, "SIMILAR PAPERS IN YOUR LIBRARY"), (similarBox = h("div", {}, h("span", { class: "muted small" }, "Looking…")))),
                h("section", { style: "margin-top:.8rem" }, h("h3", { class: "small muted" }, "MORE PAPERS ON THIS TOPIC"), (onlineBox = h("div", {}, h("span", { class: "muted small" }, "Looking…")))));
            similar(simText, similarBox);
            online(d.search_hint, title, onlineBox);
            if (paper && paper.in_library) body.append(h("div", { style: "margin-top:1rem" }, h("button", { class: "btn btn--accent", type: "button", onclick: () => PL.ask("Explain this paper in simple terms.", { id: paper.in_library, title: paper.title }) }, "Ask the assistant about this paper")));
        } catch (e) { PL.error(body, e); }
    }
    let similarBox, onlineBox;

    async function similar(text, box) {
        try {
            const d = await PL.api("/api/similar", { json: { text } });
            box.replaceChildren(d.similar.length ? d.similar.slice(0, 5).map((s) => h("div", { class: "row", style: "padding:.3rem 0;border-bottom:1px solid var(--grid)" },
                h("button", { class: "chart__toggle", style: "text-align:left;text-decoration:none;color:var(--ink-1)", type: "button", onclick: () => PL.openPaper(s.paper_id) }, s.title))) : h("span", { class: "muted small" }, "Nothing similar in your library yet."));
        } catch (_) { box.replaceChildren(h("span", { class: "muted small" }, "Not available right now.")); }
    }

    async function online(hint, title, box) {
        if (!hint) return void box.replaceChildren(h("span", { class: "muted small" }, "Not available."));
        try {
            const d = await PL.api("/api/search?limit=6&q=" + encodeURIComponent(hint));
            const rest = d.results.filter((r) => (r.title || "").toLowerCase() !== (title || "").toLowerCase()).slice(0, 5);
            box.replaceChildren(rest.length ? rest.map((r) => h("div", { style: "padding:.35rem 0;border-bottom:1px solid var(--grid)" }, h("div", { class: "small" }, r.title), h("div", { class: "paper__meta" }, [PL.fmt.authors(r.authors, 2), r.year].filter(Boolean).join(" · ")),
                h("div", { class: "row" }, h("button", { class: "btn btn--small btn--ghost", type: "button", onclick: () => analyze({ title: r.title, text: r.abstract, paper: r }) }, "Analyze"), r.url ? h("a", { class: "small", href: r.url, target: "_blank", rel: "noopener" }, "Open") : null))) : h("span", { class: "muted small" }, "No more papers found."));
        } catch (_) { box.replaceChildren(h("span", { class: "muted small" }, "Not available right now.")); }
    }

    PL.views.discover = {
        init(el) {
            input = h("input", { class: "search-input", type: "search", placeholder: "Enter a topic, e.g. graph neural networks, climate forecasting…", "aria-label": "Topic", autocomplete: "off", onkeydown: (e) => { if (e.key === "Enter") search(input.value); } });
            searchBtn = h("button", { class: "btn btn--primary", type: "button", onclick: () => search(input.value) }, "Find papers");
            resultsBox = h("div", {});
            const file = h("input", { type: "file", accept: ".pdf", id: "own-file", hidden: true });
            const text = h("textarea", { class: "input", rows: 4, style: "width:100%", placeholder: "…or paste an abstract or any text", "aria-label": "Text to analyze" });
            const fileLabel = h("span", { class: "small muted" }, "No file chosen");
            file.addEventListener("change", () => { fileLabel.textContent = file.files.length ? file.files[0].name : "No file chosen"; });
            el.append(
                h("div", { class: "card search-card", style: "max-width:820px;margin:0 auto 1.2rem" }, h("div", { class: "search-bar" }, input, searchBtn),
                    h("div", { class: "tagrow", style: "margin-top:.6rem" }, h("span", { class: "small muted" }, "Try:"), ["attention in vision", "federated learning privacy", "ride sharing", "reinforcement learning robots"].map((t) => h("button", { class: "chip", type: "button", onclick: () => { input.value = t; search(t); } }, t)))),
                resultsBox,
                h("details", { class: "sec", style: "max-width:820px;margin:1.5rem auto 0" }, h("summary", {}, "Have a paper of your own? Analyze a PDF or some text"),
                    h("div", { class: "sec__body" }, h("div", { class: "row", style: "margin:.4rem 0" }, h("button", { class: "btn btn--small btn--ghost", type: "button", onclick: () => file.click() }, "Choose a PDF"), fileLabel, file),
                        text, h("div", { style: "margin-top:.5rem" }, h("button", { class: "btn btn--accent", type: "button", onclick: () => {
                            if (file.files.length) analyze({ title: file.files[0].name.replace(/\.pdf$/i, ""), file: file.files[0] });
                            else if (text.value.trim().length >= 40) analyze({ text: text.value.trim() });
                            else PL.toast("Choose a PDF or paste a few sentences first.", "error");
                        } }, "Analyze")))));
        },
    };
})();
