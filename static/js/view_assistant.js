(function () {
    "use strict";
    const { h } = PL;
    let log, input, sendBtn, scopeBox, busy = false;

    const STARTERS_ALL = ["Explain how vision transformers work, in simple words.", "Which of my papers talk about ride sharing?", "What is federated learning and why does it matter?"];
    const STARTERS_PAPER = ["Explain this paper in simple terms.", "What problem does it solve, and why does it matter?", "What did the authors find?", "What do the technical terms mean?"];

    const scrollDown = () => { log.scrollTop = log.scrollHeight; };
    const live = (t) => String(t).replace(/\[S(\d+)\]/g, "[$1]");

    // ---------- scope ----------
    function renderScope() {
        const docs = PL.state.selectedDocs;
        scopeBox.replaceChildren(
            h("span", { class: "small muted" }, "Talking about:"),
            docs.length ? docs.map((d) => h("button", { class: "chip chip--accent", type: "button", title: "Remove", onclick: () => { PL.state.selectedDocs = docs.filter((x) => x.id !== d.id); renderScope(); } }, PL.fmt.short(d.title, 48) + " ✕"))
                : h("span", { class: "chip" }, "all papers in my library"),
            h("button", { class: "btn btn--small btn--ghost", type: "button", onclick: choose }, docs.length ? "Change papers" : "Choose a paper"),
            docs.length >= 2 ? h("button", { class: "btn btn--small btn--accent", type: "button", onclick: () => send("Compare these papers: how are their ideas and methods similar and different?") }, "Compare them") : null,
            h("button", { class: "btn btn--small btn--ghost", type: "button", onclick: reset }, "New chat"));
    }
    function reset() { PL.state.history = []; log.replaceChildren(intro()); input.focus(); }

    async function choose() {
        const box = h("div", {}, h("button", { class: "btn btn--small btn--ghost drawer__close", onclick: PL.closeDrawer }, "✕ Close"), h("h2", { style: "font-size:1.2rem" }, "Which paper do you want to talk about?"),
            h("p", { class: "small muted" }, "Pick one to ask about it specifically, or several to compare them. Leave empty to ask across your whole library."));
        const q = h("input", { class: "input", type: "search", placeholder: "filter by title", style: "width:100%;margin:.5rem 0" }), list = h("div", {});
        box.append(q, list); PL.openDrawer(box);
        let papers = [];
        try { papers = (await PL.api("/api/library/papers?limit=100")).papers; } catch (e) { list.textContent = e.message; return; }
        const draw = () => list.replaceChildren(...papers.filter((p) => p.title.toLowerCase().includes(q.value.toLowerCase())).map((p) => {
            const on = PL.state.selectedDocs.some((d) => d.id === p.paper_id);
            return h("label", { class: "row", style: "padding:.35rem 0;border-bottom:1px solid var(--grid)" }, h("input", { type: "checkbox", checked: on || null, onchange: (e) => {
                PL.state.selectedDocs = e.target.checked ? [...PL.state.selectedDocs, { id: p.paper_id, title: p.title }] : PL.state.selectedDocs.filter((d) => d.id !== p.paper_id); renderScope();
            } }), h("span", { class: "small" }, p.title));
        }));
        q.addEventListener("input", draw); draw();
    }

    function intro() {
        const one = PL.state.selectedDocs.length > 0;
        return h("div", { class: "msg msg--bot" }, h("p", {}, one ? "Hi! Ask me anything about this paper and I will explain it in simple words, with the page it comes from." : "Hi! I explain research papers in simple words. Ask me about a paper in your library, or pick a paper first to go deeper."),
            h("div", { class: "tagrow", style: "margin-top:.5rem" }, (one ? STARTERS_PAPER : STARTERS_ALL).map((s) => h("button", { class: "chip", type: "button", onclick: () => send(s) }, s))));
    }

    // ---------- sending ----------
    async function send(text) {
        text = (text || input.value).trim();
        if (!text || busy) return;
        busy = true; sendBtn.disabled = true; input.value = "";
        log.querySelectorAll(".suggest").forEach((n) => n.remove());
        log.append(h("div", { class: "msg msg--user" }, text));
        const bot = h("div", { class: "msg msg--bot", "aria-live": "polite" }, h("span", { class: "muted" }, "Reading the paper…"));
        log.append(bot); scrollDown();
        try {
            const final = await stream({ message: text, history: PL.state.history.slice(-8), paper_ids: PL.state.selectedDocs.map((d) => d.id) }, bot);
            PL.state.history.push({ role: "user", content: text, context: final.context }, { role: "assistant", content: final.answer });
            render(bot, final);
        } catch (e) { bot.replaceChildren(h("div", { role: "alert" }, "⚠ " + e.message)); }
        finally { busy = false; sendBtn.disabled = false; scrollDown(); input.focus(); }
    }

    async function stream(body, bot) {
        const resp = await fetch("/api/chat/stream", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
        if (!resp.ok) { let d = {}; try { d = await resp.json(); } catch (_) { } throw new Error(d.error || "Something went wrong. Please try again."); }
        const reader = resp.body.getReader(), dec = new TextDecoder(); let buf = "", liveEl = null, final = null;
        while (true) {
            const { value, done } = await reader.read(); if (done) break;
            buf += dec.decode(value, { stream: true });
            let i;
            while ((i = buf.indexOf("\n\n")) >= 0) {
                const line = buf.slice(0, i).trim(); buf = buf.slice(i + 2);
                if (!line.startsWith("data:")) continue;
                const ev = JSON.parse(line.slice(5));
                if (ev.type === "token") { if (!liveEl) { liveEl = h("div", {}); bot.replaceChildren(liveEl); } liveEl.textContent += live(ev.text); scrollDown(); }
                else if (ev.type === "final") final = ev;
                else if (ev.type === "error") throw new Error(ev.error);
            }
        }
        if (!final) throw new Error("The answer did not finish. Please try again.");
        return final;
    }

    // ---------- rendering ----------
    function highlighted(text, spans) {
        if (!spans || !spans.length) return [document.createTextNode(text)];
        const m = [];
        spans.slice().sort((a, b) => a.start - b.start).forEach((s) => { const l = m[m.length - 1]; if (l && s.start <= l.end) l.end = Math.max(l.end, s.end); else m.push({ start: s.start, end: s.end }); });
        const out = []; let i = 0;
        m.forEach((s) => { if (s.start > i) out.push(document.createTextNode(text.slice(i, s.start))); out.push(h("mark", { class: "hl" }, text.slice(s.start, s.end))); i = s.end; });
        if (i < text.length) out.push(document.createTextNode(text.slice(i)));
        return out;
    }

    function sentence(s, srcWrap) {
        const span = h("span", { class: "sent" + (s.kind === "interpretation" ? " sent--interp" : "") });
        String(s.text).split(/(\[\d+\])/).forEach((p) => {
            const m = p.match(/^\[(\d+)\]$/);
            if (m) span.append(h("button", { class: "cite", type: "button", "aria-label": "Show source " + m[1], onclick: () => { const d = srcWrap.querySelector("details"); if (d) d.open = true; const c = srcWrap.querySelector("#src-" + m[1]); if (c) { srcWrap.querySelectorAll(".ev").forEach((x) => x.classList.remove("is-active")); c.classList.add("is-active"); c.scrollIntoView({ behavior: "smooth", block: "nearest" }); } } }, m[1]));
            else span.append(document.createTextNode(p));
        });
        if (s.kind === "unverified") span.append(h("span", { class: "chip chip--warn", style: "margin-left:.3rem", title: "The paper does not clearly say this" }, "check this"));
        span.append(" ");
        return span;
    }

    function render(bot, r) {
        const srcWrap = h("div", {});
        const cited = r.sources.filter((s) => s.cited), shown = cited.length ? cited : r.sources;
        if (shown.length) srcWrap.append(h("details", { class: "sec", style: "margin-top:.6rem" }, h("summary", {}, `Sources (${shown.length})`), h("div", { class: "sec__body" }, shown.map((s) => {
            const text = h("div", { class: "ev__text" }, ...highlighted(s.text, s.highlights));
            return h("div", { class: "ev", id: "src-" + s.n }, h("div", { class: "ev__title" }, h("span", { class: "cite", style: "cursor:default" }, s.n), " " + s.title),
                h("div", { class: "ev__meta" }, `${s.pages[0] === s.pages[1] ? "page " + s.pages[0] : "pages " + s.pages[0] + "–" + s.pages[1]} · ${s.section.replace(/_/g, " ")}`),
                h("div", { class: "small" }, "“" + s.quote + "”"),
                h("div", { class: "row", style: "margin-top:.3rem" }, h("button", { class: "chip", type: "button", onclick: () => text.classList.toggle("is-open") }, "Show the full passage"), h("button", { class: "chip", type: "button", onclick: () => PL.openPaper(s.paper_id) }, "Open the paper")), text);
        }))));
        const body = r.kind === "none" ? h("p", {}, r.answer) : h("div", {}, r.sentences.map((s) => {
            const n = sentence(s, srcWrap);
            return s.text.trim().startsWith("•") ? h("div", { style: "margin:.4rem 0" }, n) : n;
        }));
        bot.replaceChildren(body, r.notes.map((n) => h("p", { class: "note" }, n)), srcWrap);
        if (r.suggestions && r.suggestions.length) log.append(h("div", { class: "tagrow suggest", style: "margin:.2rem 0 .6rem" }, r.suggestions.map((s) => h("button", { class: "chip", type: "button", onclick: () => send(s) }, s))));
        scrollDown();
    }

    PL.views.assistant = {
        init(el) {
            el.append(h("h2", { class: "view__title" }, "Assistant"), h("p", { class: "view__lead" }, "Ask in your own words. I answer in simple terms and show where in the paper each answer comes from."));
            scopeBox = h("div", { class: "row", style: "margin-bottom:.8rem" });
            log = h("div", { class: "chat__log", role: "log", style: "max-height:62vh" }, intro());
            input = h("textarea", { class: "input", rows: 2, placeholder: "Type your question… (Enter to send)", "aria-label": "Your question", maxlength: 1000, onkeydown: (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } } });
            sendBtn = h("button", { class: "btn btn--primary", type: "button", onclick: () => send() }, "Ask");
            el.append(scopeBox, h("div", { class: "card", style: "max-width:900px" }, log, h("div", { class: "composer" }, input, sendBtn)));
            renderScope();
        },
        show() {
            renderScope();
            const q = PL.state.pendingQuestion;
            PL.state.pendingQuestion = null;
            if (q) { if (!PL.state.history.length) log.replaceChildren(); send(q); } else input.focus();
        },
    };
})();
