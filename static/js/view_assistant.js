(function () {
    "use strict";
    const { h, esc, fmt } = PL;
    let log, srcPanel, input, sendBtn, scopeBox, busy = false, evCards = {};

    const EXAMPLES = [
        "Explain the methodology used in this paper.", "What are the major findings?", "Which datasets were used?",
        "What are the limitations of these studies?", "Which papers discuss transformer-based approaches?",
        "What research gaps can be identified across these papers?", "Summarize the evolution of attention mechanisms in computer vision.",
    ];

    // ---------- helpers ----------
    function highlighted(text, spans) {
        if (!spans || !spans.length) return [document.createTextNode(text)];
        const m = [];
        spans.slice().sort((a, b) => a.start - b.start).forEach((s) => { const l = m[m.length - 1]; if (l && s.start <= l.end) l.end = Math.max(l.end, s.end); else m.push({ start: s.start, end: s.end }); });
        const out = []; let i = 0;
        m.forEach((s) => { if (s.start > i) out.push(document.createTextNode(text.slice(i, s.start))); out.push(h("mark", { class: "hl" }, text.slice(s.start, s.end))); i = s.end; });
        if (i < text.length) out.push(document.createTextNode(text.slice(i)));
        return out;
    }
    const scrollLog = () => { log.scrollTop = log.scrollHeight; };

    function renderScope() {
        const docs = PL.state.selectedDocs;
        scopeBox.replaceChildren(
            h("span", { class: "small muted" }, "Scope: "),
            docs.length ? docs.map((d) => h("button", { class: "chip chip--accent", type: "button", title: "Remove from scope", onclick: () => { PL.state.selectedDocs = docs.filter((x) => x.id !== d.id); renderScope(); } }, PL.fmt.short(d.title, 46) + " ✕"))
                : h("span", { class: "chip" }, "all papers"),
            h("button", { class: "btn btn--small btn--ghost", type: "button", onclick: choosePapers }, "Choose papers…"),
            docs.length >= 2 ? h("button", { class: "btn btn--small btn--accent", type: "button", onclick: () => send(`Compare the methodologies of these ${docs.length} papers.`) }, "Compare selected") : null,
            h("button", { class: "btn btn--small btn--ghost", type: "button", onclick: () => { PL.state.history = []; log.replaceChildren(intro()); srcPanel.replaceChildren(emptySources()); } }, "New conversation"));
    }

    async function choosePapers() {
        const box = h("div", {}, h("button", { class: "btn btn--small btn--ghost drawer__close", onclick: PL.closeDrawer }, "✕ Close"), h("h2", { style: "font-size:1.2rem" }, "Choose papers to ask about"),
            h("p", { class: "small muted" }, "Leave empty to search all papers. Select two or more to compare them."));
        const list = h("div", {}); const q = h("input", { class: "input", type: "search", placeholder: "filter by title", style: "width:100%;margin:.5rem 0" });
        box.append(q, list); PL.openDrawer(box);
        let papers = [];
        try { papers = (await PL.api("/api/library/papers?limit=100&status=processed")).papers; } catch (e) { list.textContent = e.message; return; }
        const draw = () => {
            const term = q.value.toLowerCase();
            list.replaceChildren(...papers.filter((p) => p.title.toLowerCase().includes(term)).map((p) => {
                const on = PL.state.selectedDocs.some((d) => d.id === p.paper_id);
                return h("label", { class: "row", style: "padding:.3rem 0;border-bottom:1px solid var(--grid)" }, h("input", { type: "checkbox", checked: on || null, onchange: (e) => {
                    PL.state.selectedDocs = e.target.checked ? [...PL.state.selectedDocs, { id: p.paper_id, title: p.title }] : PL.state.selectedDocs.filter((d) => d.id !== p.paper_id); renderScope();
                } }), h("span", { class: "small" }, p.title));
            }));
        };
        q.addEventListener("input", draw); draw();
    }

    function intro() {
        return h("div", { class: "msg msg--bot" }, h("p", {}, "Ask a question about the indexed papers. Answers cite the passages they rely on, show the page and section, and flag statements the sources do not support."),
            h("p", { class: "small muted" }, "Try one:"), h("div", { class: "tagrow" }, EXAMPLES.map((e) => h("button", { class: "chip", type: "button", onclick: () => send(e) }, e))));
    }
    const emptySources = () => h("div", { class: "empty" }, "Evidence for the latest answer appears here.");

    // ---------- sending ----------
    async function send(text) {
        text = (text || input.value).trim();
        if (!text || busy) return;
        busy = true; sendBtn.disabled = true; input.value = "";
        log.append(h("div", { class: "msg msg--user" }, text));
        const bot = h("div", { class: "msg msg--bot", "aria-live": "polite" }, h("span", { class: "muted" }, "Retrieving evidence…"));
        log.append(bot); scrollLog();
        const body = { question: text, history: PL.state.history.slice(-6), document_ids: PL.state.selectedDocs.map((d) => d.id) };
        try {
            const final = await stream(body, bot);
            PL.state.history.push({ role: "user", content: text, retrieval_query: final.query.retrieval_query }, { role: "assistant", content: final.answer });
            PL.state.lastResponse = final;
            let expl = null;
            try { expl = await PL.api("/api/explain/response", { json: { response: final } }); } catch (_) { /* answer is still shown, without highlights */ }
            renderAnswer(bot, final, expl);
        } catch (e) {
            bot.replaceChildren(h("div", { role: "alert" }, "⚠ " + (e.message || "Request failed") + (e.status === 503 ? " — retrieval is unavailable (primary and fallback both down)." : "")));
        } finally { busy = false; sendBtn.disabled = false; scrollLog(); input.focus(); }
    }

    async function stream(body, bot) {
        const resp = await fetch("/api/rag/stream", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
        if (!resp.ok) { let d = {}; try { d = await resp.json(); } catch (_) { } const e = new Error(d.error || resp.statusText); e.status = resp.status; throw e; }
        const reader = resp.body.getReader(), dec = new TextDecoder(); let buf = "", live = null, final = null;
        while (true) {
            const { value, done } = await reader.read(); if (done) break;
            buf += dec.decode(value, { stream: true });
            let idx;
            while ((idx = buf.indexOf("\n\n")) >= 0) {
                const line = buf.slice(0, idx).trim(); buf = buf.slice(idx + 2);
                if (!line.startsWith("data:")) continue;
                const ev = JSON.parse(line.slice(5));
                if (ev.type === "meta") { live = h("div", { class: "pre" }); bot.replaceChildren(h("div", { class: "small muted" }, `Retrieved ${ev.sources.length} passages via ${PL.modelLabel(ev.retriever)}${ev.fallback_active ? " (fallback)" : ""}. Generating…`), live); }
                else if (ev.type === "token" && live) { live.textContent += ev.text; scrollLog(); }
                else if (ev.type === "final") final = ev;
                else if (ev.type === "error") throw new Error("The answer stream failed (" + ev.error + ")");
            }
        }
        if (!final) throw new Error("The stream ended without a final answer");
        return final;
    }

    // ---------- rendering ----------
    function sentenceNode(s) {
        const kind = s.kind || (s.status === "interpretation" ? "interpretation" : "stated_in_source");
        const cls = "sent" + (s.status === "weak" ? " sent--weak" : "") + (kind === "unverified" ? " sent--unverified" : "") + (kind === "interpretation" ? " sent--interp" : "");
        const parts = String(s.text).split(/(\[\s*S\d+(?:\s*[,;]\s*S?\d+)*\s*\])/);
        const span = h("span", { class: cls, title: s.warning || (s.evidence_strength ? "Evidence strength: " + s.evidence_strength + " (lexical, uncalibrated)" : "") });
        parts.forEach((p) => {
            const m = p.match(/^\[\s*(S\d+(?:\s*[,;]\s*S?\d+)*)\s*\]$/);
            if (m) m[1].split(/[,;]/).forEach((l) => { l = l.trim(); l = l.startsWith("S") ? l : "S" + l; span.append(h("button", { class: "cite", type: "button", "aria-label": "Show source " + l, onclick: () => focusEvidence(l) }, l)); });
            else span.append(document.createTextNode(p));
        });
        if (kind === "unverified") span.append(h("span", { class: "chip chip--bad", style: "margin-left:.3rem" }, "⚠ unverified"));
        if (kind === "interpretation") span.append(h("span", { class: "chip", style: "margin-left:.3rem" }, "interpretation"));
        if (s.evidence_strength && kind === "stated_in_source") span.append(h("span", { class: "strength pill-" + s.evidence_strength, title: "Lexical-overlap indicator, not a probability" }, " ● " + s.evidence_strength));
        span.append(" ");
        return span;
    }

    function renderAnswer(bot, r, expl) {
        const sents = expl ? expl.sentences : r.sentences.map((s) => Object.assign({ kind: s.status === "interpretation" ? "interpretation" : "stated_in_source" }, s));
        const modeLabel = { llm: "Language-model answer", extractive: "Extractive passages (no language model answer available)", insufficient: "Insufficient evidence" }[r.answer_mode] || r.answer_mode;
        const ver = r.verification || {};
        bot.replaceChildren(
            h("div", { class: "row", style: "margin-bottom:.4rem" },
                h("span", { class: "chip " + (r.answer_mode === "llm" ? "chip--ok" : r.answer_mode === "insufficient" ? "chip--bad" : "chip--warn") }, modeLabel),
                h("span", { class: "chip" }, `${r.query.intent} · ${r.query.scope.replace("_", "-")}`),
                r.retrieval.fallback_active ? h("span", { class: "chip chip--warn" }, "⚠ fallback retrieval (TF-IDF)") : h("span", { class: "chip" }, PL.modelLabel(r.retrieval.retriever)),
                r.query.is_followup ? h("span", { class: "chip" }, "follow-up") : null),
            ver.replaced ? h("div", { class: "banner banner--info", style: "margin-bottom:.5rem" }, h("span", {}, "The language model's draft was rejected (" + ver.reason + "), so only passages quoted from the papers are shown. "),
                h("details", {}, h("summary", { class: "small" }, "Show the rejected draft (unverified)"), h("div", { class: "pre" }, ver.llm_answer))) : null,
            r.answer_mode === "insufficient" ? h("p", {}, r.answer) : h("div", {}, sents.map((s) => {
                const t = String(s.text).trim();
                if (s.kind === "meta") return h("div", { class: "small muted", style: "margin-bottom:.3rem" }, t);
                const node = sentenceNode(s);
                return t.startsWith("•") ? h("div", { style: "margin:.35rem 0;padding-left:.2rem" }, node) : node;
            })),
            r.query.warnings.length ? h("p", { class: "note" }, r.query.warnings.join(" ")) : null,
            expl && expl.papers && Object.keys(expl.papers).length > 1 ? paperShares(expl.papers) : null,
            expl && expl.unsupported_claims.length ? h("details", { class: "sec" }, h("summary", {}, `⚠ ${expl.unsupported_claims.length} statement(s) not backed by a retrieved passage`), h("div", { class: "sec__body" }, expl.unsupported_claims.map((u) => h("p", { class: "small" }, u)))) : null,
            h("div", { class: "msg__meta" }, metaLine(r)));
        renderSources(r, expl);
        scrollLog();
    }

    function metaLine(r) {
        const t = r.timings, g = r.generation;
        return [`retrieval ${fmt.secs(t.retrieval_s)}`, g.provider && r.answer_mode === "llm" ? `generation ${fmt.secs(t.generation_s)} (${g.provider}${g.model ? ", " + g.model : ""})` : null,
            g.prompt_tokens ? `${g.prompt_tokens}+${g.completion_tokens} tokens` : null, g.cost_usd != null ? `$${g.cost_usd.toFixed(4)}` : null,
            r.evidence.signal != null ? `evidence ${r.evidence.gate.replace("_", " ")} ${r.evidence.signal}${r.evidence.threshold ? " (threshold " + r.evidence.threshold + ")" : ""}` : null,
            (g.provider_errors || []).length ? "notes: " + g.provider_errors.map((e) => PL.fmt.short(e, 70)).join(" | ") : null].filter(Boolean).join(" · ");
    }

    function paperShares(papers) {
        const wrap = h("div", { style: "margin-top:.6rem" }, h("div", { class: "small muted" }, "Contribution of each paper to the cited claims"));
        PL.charts.hbars(wrap, { title: "", format: (v) => fmt.pct(v), max: 1, valueName: "share of cited claims",
            items: Object.values(papers).map((p, i) => ({ label: PL.fmt.short(p.title, 40), value: p.share_of_cited_claims, color: `var(--viz-${["tfidf", "sbert", "e5", "bge", "hybrid"][i % 5]})` })) });
        return wrap;
    }

    function renderSources(r, expl) {
        evCards = {};
        const spansBy = {}, strength = {};
        if (expl) expl.sentences.forEach((s) => s.evidence.forEach((e) => { (spansBy[e.label] = spansBy[e.label] || []).push(...e.highlights); if (e.supporting) strength[e.label] = s.evidence_strength; }));
        srcPanel.replaceChildren(h("h3", {}, "Evidence ", h("small", {}, `${r.references.length} passages`)),
            ...r.references.map((ref) => {
                const text = h("div", { class: "ev__text" }, ...highlighted(ref.text, spansBy[ref.label]));
                const card = h("div", { class: "ev", id: "ev-" + ref.label },
                    h("div", { class: "ev__title" }, h("span", { class: "cite", style: "cursor:default" }, ref.label), " " + ref.title),
                    h("div", { class: "ev__meta" }, `${ref.pages[0] === ref.pages[1] ? "p. " + ref.pages[0] : "pp. " + ref.pages[0] + "–" + ref.pages[1]} · ${ref.section.replace(/_/g, " ")} · retrieval rank ${ref.retrieval_rank}, score ${ref.retrieval_score}`,
                        ref.cited ? [" · ", h("span", { class: "chip chip--ok" }, "cited"), strength[ref.label] ? h("span", { class: "strength pill-" + strength[ref.label] }, " " + strength[ref.label]) : null] : [" · ", h("span", { class: "chip" }, "retrieved, not cited")]),
                    text,
                    h("div", { class: "row", style: "margin-top:.3rem" }, h("button", { class: "chip", type: "button", onclick: () => { text.classList.toggle("is-open"); } }, "Show full passage"),
                        ref.url ? h("a", { class: "chip", href: ref.url, target: "_blank", rel: "noopener" }, "source link") : null,
                        ref.paper_id ? h("button", { class: "chip", type: "button", onclick: () => PL.openPaper && PL.openPaper(ref.paper_id) }, "open paper") : null));
                evCards[ref.label] = card;
                return card;
            }),
            h("p", { class: "note" }, "Highlights mark terms and numbers shared with the citing sentence. Evidence strength is a lexical-overlap indicator, not a calibrated probability."));
    }

    function focusEvidence(label) {
        const c = evCards[label]; if (!c) return;
        Object.values(evCards).forEach((x) => x.classList.remove("is-active")); c.classList.add("is-active"); c.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    PL.views.assistant = {
        init(el) {
            el.append(h("h2", { class: "view__title" }, "AI research assistant"), h("p", { class: "view__lead" }, "Conversational Q&A grounded in the indexed papers, with citations, page numbers and evidence you can inspect."));
            scopeBox = h("div", { class: "row", style: "margin-bottom:.8rem" });
            log = h("div", { class: "chat__log", role: "log" }, intro());
            input = h("textarea", { class: "input", rows: 2, placeholder: "Ask about one paper, or across papers… (Enter to send, Shift+Enter for a new line)", "aria-label": "Your question", maxlength: 1000, onkeydown: (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } } });
            sendBtn = h("button", { class: "btn btn--primary", type: "button", onclick: () => send() }, "Ask");
            srcPanel = h("div", { class: "card", style: "max-height:76vh;overflow-y:auto" }, emptySources());
            el.append(scopeBox, h("div", { class: "chat" }, h("div", { class: "card" }, log, h("div", { class: "composer" }, input, sendBtn)), srcPanel));
            renderScope();
        },
        show() { renderScope(); },
    };
})();
