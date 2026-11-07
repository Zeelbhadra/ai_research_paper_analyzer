(function () {
    "use strict";
    const { h, fmt } = PL;
    let root, grid, countEl, pager, f = { q: "", author: "", topic: "", year_from: "", year_to: "", status: "processed" }, offset = 0;
    const LIMIT = 24;

    async function load() {
        PL.loading(grid);
        const qs = new URLSearchParams({ ...f, limit: LIMIT, offset });
        for (const [k, v] of [...qs]) if (v === "") qs.delete(k);
        try {
            const d = await PL.api("/api/library/papers?" + qs);
            countEl.textContent = `${d.total} paper${d.total === 1 ? "" : "s"}`;
            grid.replaceChildren(...(d.papers.length ? d.papers.map(card) : [h("div", { class: "empty" }, "No papers match these filters.")]));
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
            h("div", { class: "paper__meta" }, [p.authors.length ? p.authors.slice(0, 3).join(", ") + (p.authors.length > 3 ? " et al." : "") : "authors unknown", p.year ? String(p.year) : "year unknown", p.source].join(" · ")),
            p.summary ? h("div", { class: "paper__sum" }, p.summary) : null,
            h("div", { class: "tagrow" }, p.topic ? h("span", { class: "chip chip--accent" }, p.topic) : null, (p.keywords || []).slice(0, 3).map((k) => h("span", { class: "chip" }, k)),
                h("span", { class: "chip" }, p.n_chunks + " chunks"), p.status !== "processed" ? h("span", { class: "chip chip--warn" }, p.status) : null));
    }

    async function detail(id) {
        PL.openDrawer(h("div", { class: "loading" }, "Loading paper…"));
        try {
            const p = await PL.api("/api/library/paper?id=" + encodeURIComponent(id));
            const e = p.enrichment || {};
            const node = h("div", {},
                h("button", { class: "btn btn--small btn--ghost drawer__close", onclick: PL.closeDrawer, "aria-label": "Close" }, "✕ Close"),
                h("h2", { style: "font-size:1.2rem;margin:.3rem 0 .4rem" }, p.title),
                h("div", { class: "paper__meta" }, [p.authors.length ? p.authors.join(", ") : "authors unknown", p.published || "date unknown", p.venue, p.source].filter(Boolean).join(" · ")),
                h("div", { class: "row", style: "margin:.6rem 0" },
                    p.abs_url ? h("a", { href: p.abs_url, target: "_blank", rel: "noopener", class: "btn btn--small btn--ghost" }, "Abstract page") : null,
                    p.pdf_url ? h("a", { href: p.pdf_url, target: "_blank", rel: "noopener", class: "btn btn--small btn--ghost" }, "PDF") : null,
                    h("button", { class: "btn btn--small btn--primary", onclick: () => { PL.state.selectedDocs = [{ id: p.paper_id, title: p.title }]; PL.closeDrawer(); location.hash = "#/assistant"; } }, "Ask the assistant about this paper"),
                    p.citation_count != null ? h("span", { class: "chip" }, p.citation_count + " citations") : null, p.doi ? h("span", { class: "chip" }, "DOI " + p.doi) : null),
                p.abstract ? h("section", {}, h("h3", { class: "small muted" }, "ABSTRACT"), h("p", { class: "small" }, p.abstract)) : null,
                h("section", {}, h("h3", { class: "small muted" }, "SUMMARY ", h("span", { class: "chip" }, e.summary_method || "n/a")), h("p", {}, e.summary || "No summary generated yet."),
                    h("p", { class: "note" }, "Extractive summary (most informative sentences of the abstract/introduction), not abstractive text.")),
                h("section", {}, h("h3", { class: "small muted" }, "KEYWORDS"), h("div", { class: "tagrow" }, (e.keywords || []).map(([k, s]) => h("span", { class: "chip", title: "KeyBERT score " + s }, k)), (e.keywords || []).length ? null : h("span", { class: "muted small" }, "none"))),
                h("section", {}, h("h3", { class: "small muted" }, "TOPIC CLASSIFICATION (zero-shot)"),
                    (e.topics || []).length ? e.topics.map((t) => h("div", { class: "hbar" }, h("span", { class: "hbar__label" }, t.label), h("span", { class: "hbar__track" }, h("span", { class: "hbar__fill", style: `display:block;width:${t.score * 100}%;background:var(--viz-tfidf)` })), h("span", { class: "hbar__val" }, fmt.pct(t.score)))) : h("span", { class: "muted small" }, "not classified"),
                    h("p", { class: "note" }, "Labels come from an NLI zero-shot model over a fixed taxonomy and have not been validated against human labels; treat them as hints.")),
                h("section", {}, h("h3", { class: "small muted" }, `EXTRACTED CONTENT (${p.n_chunks} chunks, page numbers preserved)`),
                    p.sections.map((s) => h("details", { class: "sec" }, h("summary", {}, `${s.name.replace(/_/g, " ")} `, h("span", { class: "muted small" }, `(${s.chunks.length} chunk${s.chunks.length === 1 ? "" : "s"}, pp. ${Math.min(...s.chunks.map((c) => c.pages[0]))}–${Math.max(...s.chunks.map((c) => c.pages[1]))})`)),
                        h("div", { class: "sec__body" }, s.chunks.map((c) => h("div", { class: "chunk" }, h("span", { class: "chunk__pages" }, c.pages[0] === c.pages[1] ? "p. " + c.pages[0] : `pp. ${c.pages[0]}–${c.pages[1]}`), c.text)))))));
            PL.openDrawer(node);
        } catch (e) { PL.openDrawer(h("div", {}, h("button", { class: "btn btn--small btn--ghost drawer__close", onclick: PL.closeDrawer }, "✕ Close"), h("p", { role: "alert" }, "Could not load paper: " + e.message))); }
    }

    function select(label, key, options) {
        const sel = h("select", { class: "input", "aria-label": label, onchange: (e) => { f[key] = e.target.value; offset = 0; load(); } }, h("option", { value: "" }, "All"), options.map(([v, n]) => h("option", { value: v }, n ? `${v} (${n})` : v)));
        return h("label", { class: "field" }, label, sel);
    }

    PL.openPaper = detail;                       // used by the assistant's "open paper" buttons

    PL.views.library = {
        async init(el) {
            root = el;
            el.append(h("h2", { class: "view__title" }, "Research library"), h("p", { class: "view__lead" }, "Browse the indexed papers: metadata, extracted content by section and page, generated summaries, keywords and topics."));
            countEl = h("span", { class: "badge" }); grid = h("div", { class: "grid grid--3" }); pager = h("div", { class: "row row--between", style: "margin-top:1rem" });
            const bar = h("div", { class: "filters" });
            el.append(bar, h("div", { class: "row", style: "margin-bottom:.6rem" }, countEl), grid, pager);
            let facets = { authors: [], topics: [], years: [] };
            try { facets = await PL.api("/api/library/facets"); } catch (_) { /* filters degrade to free text */ }
            const years = facets.years.map(([y]) => y);
            const search = h("input", { class: "input", type: "search", placeholder: "title, abstract, author, keyword", "aria-label": "Search text", oninput: (e) => { clearTimeout(search._t); search._t = setTimeout(() => { f.q = e.target.value; offset = 0; load(); }, 300); } });
            bar.append(h("label", { class: "field" }, "Search", search), select("Author", "author", facets.authors.map(([a, n]) => [a, n])), select("Topic", "topic", facets.topics),
                h("label", { class: "field" }, "Year from", h("input", { class: "input", type: "number", min: years[0] || 1990, max: years[years.length - 1] || 2100, placeholder: years[0] || "", onchange: (e) => { f.year_from = e.target.value; offset = 0; load(); } })),
                h("label", { class: "field" }, "Year to", h("input", { class: "input", type: "number", placeholder: years[years.length - 1] || "", onchange: (e) => { f.year_to = e.target.value; offset = 0; load(); } })),
                h("label", { class: "field" }, "Status", h("select", { class: "input", onchange: (e) => { f.status = e.target.value; offset = 0; load(); } },
                    [["processed", "processed"], ["", "all"], ["failed", "failed"], ["discovered", "discovered"], ["downloaded", "downloaded"]].map(([v, n]) => h("option", { value: v }, n)))));
            if (!facets.years.length) bar.append(h("p", { class: "note", style: "grid-column:1/-1" }, "Year and author filters need publication metadata. Papers that were added as plain PDF files carry none; papers found through arXiv / Semantic Scholar do."));
        },
        show() { load(); },
    };
})();
