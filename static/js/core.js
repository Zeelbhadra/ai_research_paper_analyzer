/* Core: API helper, safe DOM helpers, formatting, router, status banner. Everything dynamic is escaped or set via
   textContent; innerHTML is only used with strings built from esc(). */
/* Guard: Element.append/prepend/replaceChildren turn null/undefined/false into the literal text "null"/"false". The views pass
   conditional children (cond ? node : null) and nested arrays, so make these methods skip empty values and flatten arrays. */
(function () {
    const clean = (args) => args.flat(Infinity).filter((k) => k != null && k !== false);
    for (const m of ["append", "prepend", "replaceChildren"]) {
        const orig = Element.prototype[m];
        Element.prototype[m] = function (...args) { return orig.apply(this, clean(args)); };
    }
})();

window.PL = (function () {
    "use strict";
    const PL = { views: {}, state: { selectedDocs: [], history: [], lastResponse: null } };

    PL.esc = function (s) {
        return String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
    };
    PL.h = function (tag, attrs, ...kids) {           // tiny element builder (safe: text goes through createTextNode)
        const el = document.createElement(tag);
        for (const [k, v] of Object.entries(attrs || {})) {
            if (v == null || v === false) continue;
            if (k === "class") el.className = v;
            else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
            else if (k === "html") el.innerHTML = v;
            else el.setAttribute(k, v === true ? "" : v);
        }
        for (const kid of kids.flat()) if (kid != null && kid !== false) el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
        return el;
    };
    PL.fmt = {
        n: (v, d = 3) => (v == null || Number.isNaN(v) ? "n/a" : Number(v).toFixed(d)),
        pct: (v, d = 0) => (v == null ? "n/a" : (v * 100).toFixed(d) + "%"),
        int: (v) => (v == null ? "n/a" : Number(v).toLocaleString()),
        ms: (v) => (v == null ? "n/a" : (v >= 100 ? v.toFixed(0) : v.toFixed(1)) + " ms"),
        mb: (v) => (v == null ? "n/a" : v >= 1024 ? (v / 1024).toFixed(1) + " GB" : v.toFixed(0) + " MB"),
        secs: (v) => (v == null ? "n/a" : v < 1 ? (v * 1000).toFixed(0) + " ms" : v >= 120 ? (v / 60).toFixed(1) + " min" : v.toFixed(v < 10 ? 1 : 0) + " s"),
        date: (ts) => { const d = typeof ts === "number" ? new Date(ts * 1000) : new Date(ts); return isNaN(d) ? "n/a" : d.toLocaleString(); },
        short: (s, n = 90) => (s && s.length > n ? s.slice(0, n - 1) + "…" : s || ""),
    };
    PL.modelLabel = (name) => (name || "").startsWith("hybrid") ? "Hybrid TF-IDF+" + (name.split(":")[1] || "dense").toUpperCase() :
        ({ tfidf: "TF-IDF", sbert: "SBERT", e5: "E5", bge: "BGE" }[name] || name);
    PL.modelColor = (name) => { const k = (name || "").startsWith("hybrid") ? "hybrid" : name; return `var(--viz-${k}, #94a3b8)`; };
    PL.toast = function (msg, type) {
        const box = document.getElementById("toast-container");
        if (!box) return;
        const el = PL.h("div", { class: "toast toast--" + (type || "info"), role: "status" }, msg);
        box.append(el);
        setTimeout(() => el.remove(), 5000);
    };

    PL.api = async function (path, opts) {
        const o = Object.assign({ headers: {} }, opts || {});
        if (o.json !== undefined) { o.method = o.method || "POST"; o.headers["Content-Type"] = "application/json"; o.body = JSON.stringify(o.json); }
        let resp;
        try { resp = await fetch(path, o); } catch (e) { throw new Error("Network error: is the server running?"); }
        let data = null;
        try { data = await resp.json(); } catch (_) { /* non-JSON */ }
        if (!resp.ok) {
            const err = new Error((data && (data.error || data.detail)) || resp.status + " " + resp.statusText);
            err.status = resp.status; err.data = data;
            throw err;
        }
        return data;
    };

    PL.loading = (el, text) => { el.replaceChildren(PL.h("div", { class: "loading" }, text || "Loading…")); };
    PL.error = (el, e, retry) => {
        el.replaceChildren(PL.h("div", { class: "empty", role: "alert" }, "Could not load this section: " + e.message + " ",
            retry ? PL.h("button", { class: "btn btn--small btn--ghost", onclick: retry }, "Retry") : null));
    };

    // ---- tooltip (shared) ----
    PL.tip = {
        show(evt, html) { const t = document.getElementById("tooltip"); t.innerHTML = html; t.hidden = false; PL.tip.move(evt); },
        move(evt) { const t = document.getElementById("tooltip"); const x = Math.min(evt.clientX + 14, window.innerWidth - t.offsetWidth - 8); t.style.left = Math.max(8, x) + "px"; t.style.top = (evt.clientY + 14) + "px"; },
        hide() { document.getElementById("tooltip").hidden = true; },
    };

    // ---- router ----
    const ROUTES = ["discover", "dashboard", "library", "assistant", "benchmarks", "explain", "pipeline"];
    let timer = null;
    PL.route = function () {
        const name = (location.hash.replace(/^#\/?/, "").split("?")[0]) || "discover";
        const route = ROUTES.includes(name) ? name : "discover";
        document.querySelectorAll(".view").forEach((v) => { v.hidden = v.dataset.view !== route; });
        document.querySelectorAll(".header__link").forEach((a) => { if (a.dataset.route === route) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current"); });
        const hero = document.getElementById("hero-section"); if (hero) hero.hidden = route !== "discover";
        clearInterval(timer); timer = null;
        const v = PL.views[route];
        if (v) {
            const el = document.getElementById("view-" + route);
            if (!el.dataset.ready) { el.dataset.ready = "1"; v.init(el); }
            if (v.show) v.show(el);
            if (v.refreshEvery) timer = setInterval(() => { if (!document.hidden) v.show(el); }, v.refreshEvery);
        }
        document.title = "PaperLens — " + route.charAt(0).toUpperCase() + route.slice(1);
        window.scrollTo(0, 0);
    };

    // ---- status banner: shows when retrieval is in fallback mode (polls the cheap dashboard endpoint, never forces a model load) ----
    PL.pollStatus = async function () {
        const b = document.getElementById("status-banner");
        try {
            const d = await PL.api("/api/dashboard");
            const f = d.fallback;
            if (f && f.fallback_active) {
                b.className = "banner"; b.hidden = false;
                b.replaceChildren(PL.h("strong", {}, "⚠ Fallback retrieval active"), PL.h("span", {}, " — answers use " + PL.modelLabel(f.fallback || "tfidf") + " instead of " + PL.modelLabel(f.primary) + (f.reason ? " (" + PL.fmt.short(f.reason, 110) + ")" : "") + ". The system recovers automatically when the primary model is healthy."));
            } else b.hidden = true;
        } catch (_) { /* leave banner unchanged */ }
    };

    PL.start = function () {
        window.addEventListener("hashchange", PL.route);
        document.addEventListener("keydown", (e) => { if (e.key === "Escape") PL.closeDrawer(); });
        PL.route();
        PL.pollStatus();
        setInterval(() => { if (!document.hidden) PL.pollStatus(); }, 20000);
    };

    PL.openDrawer = (node) => { const d = document.getElementById("drawer"); d.replaceChildren(node); d.hidden = false; d.scrollTop = 0; const c = d.querySelector(".drawer__close"); if (c) c.focus(); };
    PL.closeDrawer = () => { document.getElementById("drawer").hidden = true; };
    return PL;
})();
