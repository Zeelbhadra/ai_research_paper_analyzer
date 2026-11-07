(function () {
    "use strict";
    const { h } = PL;
    let grid, countEl, pager, f = { q: "", author: "", topic: "", year_from: "", year_to: "" }, offset = 0;
    const LIMIT = 24;

    const EXPLAIN = [
        ["Explain it simply", "Explain this paper in simple terms."],
        ["Why does it matter?", "What problem does this paper solve, and why does it matter?"],
        ["Key ideas", "What are the key ideas of this paper, in simple words?"],
        ["How does it work?", "How does the method in this paper work? Explain it step by step in simple words."],
        ["What did they find?", "What did the authors find, in simple words?"],
        ["Explain the jargon", "What do the technical terms in this paper mean?"],
    ];

    async function load() {
        PL.loading(grid);
        const qs = new URLSearchParams({ ...f, limit: LIMIT, offset });
        for (const [k, v] of [...qs]) if (v === "") qs.delete(k);
        try {
            const d = await PL.api("/api/library/papers?" + qs);
            countEl.textContent = `${d.total} paper${d.total === 1 ? "" : "s"}`;
            grid.replaceChildren(...(d.papers.length ? d.papers.map(card) : [h("div", { class: "empty" }, "No papers match. Add some from Discover, or clear the filters.")]));
            pager.replaceChildren(
                h("button", { class: "btn btn--small btn--ghost", disabled: offset === 0, onclick: () => { offset = Math.max(0, offset - LIMIT); load(); } }, "← Previous"),
                h("span", { class: "muted small" }, d.total ? `${offset + 1}–${Math.min(offset + LIMIT, d.total)} of ${d.total}` : ""),
                h("button", { class: "btn btn--small btn--ghost", disabled: offset + LIMIT >= d.total, onclick: () => { offset += LIMIT; load(); } }, "Next →"));
        } catch (e) { PL.error(grid, e, load); }
    }

    function card(p) {
        const open = () => detail(p.paper_id);
        return h("article", { class: "card paper", tabindex: "0", role: "button", "aria-label": "Open " + p.title, onclick: open, onkeydown: (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); } } },
            h("div", { class: "paper__title" }, p.title || "(untitled)"),
            h("div", { class: "paper__meta" }, [PL.fmt.authors(p.authors), p.year, p.venue].filter(Boolean).join(" · ")),
            p.summary ? h("div", { class: "paper__sum" }, p.summary) : null,
            h("div", { class: "tagrow" }, p.topic ? h("span", { class: "chip chip--accent" }, p.topic) : null, (p.keywords || []).slice(0, 3).map((k) => h("span", { class: "chip" }, k))));
    }

    async function detail(id) {
        PL.openDrawer(h("div", { class: "loading" }, "Loading paper…"));
        try {
            const p = await PL.api("/api/library/paper?id=" + encodeURIComponent(id));
            const ask = (q) => PL.ask(q, { id: p.paper_id, title: p.title });
            PL.openDrawer(h("div", {},
                h("button", { class: "btn btn--small btn--ghost drawer__close", onclick: PL.closeDrawer, "aria-label": "Close" }, "✕ Close"),
                h("h2", { style: "font-size:1.2rem;margin:.3rem 0 .4rem" }, p.title),
                h("div", { class: "paper__meta" }, [p.authors.length ? p.authors.join(", ") : null, p.year, p.venue].filter(Boolean).join(" · ")),
                h("div", { class: "row", style: "margin:.6rem 0" },
                    p.abs_url ? h("a", { href: p.abs_url, target: "_blank", rel: "noopener", class: "btn btn--small btn--ghost" }, "Open the paper") : null,
                    p.pdf_url ? h("a", { href: p.pdf_url, target: "_blank", rel: "noopener", class: "btn btn--small btn--ghost" }, "PDF") : null,
                    p.citation_count != null ? h("span", { class: "chip" }, p.citation_count + " citations") : null),
                h("section", { class: "card", style: "margin:.6rem 0" }, h("h3", {}, "Understand this paper"),
                    h("p", { class: "small muted", style: "margin-bottom:.5rem" }, "The assistant explains it in simple words, using the paper itself:"),
                    h("div", { class: "tagrow" }, EXPLAIN.map(([label, q]) => h("button", { class: "chip chip--accent", type: "button", onclick: () => ask(q) }, label)),
                        h("button", { class: "chip", type: "button", onclick: () => ask(null) }, "Ask your own question"))),
                p.summary ? h("section", {}, h("h3", { class: "small muted" }, "SUMMARY"), h("p", {}, p.summary)) : null,
                p.keywords.length ? h("section", {}, h("h3", { class: "small muted" }, "KEYWORDS"), h("div", { class: "tagrow" }, p.keywords.map((k) => h("span", { class: "chip" }, k)))) : null,
                p.topics.length ? h("section", { style: "margin-top:.8rem" }, h("h3", { class: "small muted" }, "RELATED TOPICS"), h("div", { class: "tagrow" }, p.topics.slice(0, 4).map((t) => h("span", { class: "chip chip--accent" }, t.label)))) : null,
                h("section", { style: "margin-top:.8rem" }, h("h3", { class: "small muted" }, "READ THE PAPER"),
                    p.sections.map((s) => h("details", { class: "sec" }, h("summary", {}, s.name.replace(/_/g, " "), " ", h("span", { class: "muted small" }, `(pp. ${Math.min(...s.chunks.map((c) => c.pages[0]))}–${Math.max(...s.chunks.map((c) => c.pages[1]))})`)),
                        h("div", { class: "sec__body" }, s.chunks.map((c) => h("div", { class: "chunk" }, h("span", { class: "chunk__pages" }, c.pages[0] === c.pages[1] ? "page " + c.pages[0] : `pages ${c.pages[0]}–${c.pages[1]}`), c.text))))))));
        } catch (e) { PL.openDrawer(h("div", {}, h("button", { class: "btn btn--small btn--ghost drawer__close", onclick: PL.closeDrawer }, "✕ Close"), h("p", { role: "alert" }, e.message))); }
    }
    PL.openPaper = detail;

    function select(label, key, options) {
        return h("label", { class: "field" }, label, h("select", { class: "input", "aria-label": label, onchange: (e) => { f[key] = e.target.value; offset = 0; load(); } },
            h("option", { value: "" }, "All"), options.map(([v]) => h("option", { value: v }, v))));
    }

    PL.views.library = {
        async init(el) {
            el.append(h("h2", { class: "view__title" }, "Your library"), h("p", { class: "view__lead" }, "Papers you can read and ask the assistant about. Add more from Discover."));
            countEl = h("span", { class: "badge" }); grid = h("div", { class: "grid grid--3" }); pager = h("div", { class: "row row--between", style: "margin-top:1rem" });
            const bar = h("div", { class: "filters" });
            el.append(bar, h("div", { class: "row", style: "margin-bottom:.6rem" }, countEl), grid, pager);
            let facets = { authors: [], topics: [], years: [] };
            try { facets = await PL.api("/api/library/facets"); } catch (_) { /* filters degrade to text search */ }
            const search = h("input", { class: "input", type: "search", placeholder: "title, topic, author, keyword", "aria-label": "Search your library", oninput: (e) => { clearTimeout(search._t); search._t = setTimeout(() => { f.q = e.target.value; offset = 0; load(); }, 300); } });
            bar.append(h("label", { class: "field" }, "Search", search), facets.topics.length ? select("Topic", "topic", facets.topics) : null, facets.authors.length ? select("Author", "author", facets.authors) : null,
                facets.years.length ? h("label", { class: "field" }, "Year from", h("input", { class: "input", type: "number", placeholder: facets.years[0][0], onchange: (e) => { f.year_from = e.target.value; offset = 0; load(); } })) : null,
                facets.years.length ? h("label", { class: "field" }, "Year to", h("input", { class: "input", type: "number", placeholder: facets.years[facets.years.length - 1][0], onchange: (e) => { f.year_to = e.target.value; offset = 0; load(); } })) : null);
        },
        show() { load(); },
    };
})();
