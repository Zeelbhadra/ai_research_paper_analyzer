/* Core: API helper, safe DOM helpers, formatting, router. Everything dynamic is escaped or set via textContent. */

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
    const PL = { views: {}, state: { selectedDocs: [], history: [], pendingQuestion: null } };

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
        pct: (v, d = 0) => (v == null ? "" : (v * 100).toFixed(d) + "%"),
        short: (s, n = 90) => (s && s.length > n ? s.slice(0, n - 1) + "…" : s || ""),
        authors: (a, n = 3) => (a && a.length ? a.slice(0, n).join(", ") + (a.length > n ? " et al." : "") : ""),
    };
    PL.toast = function (msg, type) {
        const box = document.getElementById("toast-container");
        if (!box) return;
        const el = PL.h("div", { class: "toast toast--" + (type || "info"), role: "status" }, msg);
        box.append(el);
        setTimeout(() => el.remove(), 6000);
    };

    PL.api = async function (path, opts) {
        const o = Object.assign({ headers: {} }, opts || {});
        if (o.json !== undefined) { o.method = o.method || "POST"; o.headers["Content-Type"] = "application/json"; o.body = JSON.stringify(o.json); }
        let resp;
        try { resp = await fetch(path, o); } catch (e) { throw new Error("Could not reach the server. Please check your connection and try again."); }
        let data = null;
        try { data = await resp.json(); } catch (_) { /* non-JSON */ }
        if (!resp.ok) {
            const err = new Error((data && (data.error || data.detail)) || "Something went wrong. Please try again.");
            err.status = resp.status; err.data = data;
            throw err;
        }
        return data;
    };

    PL.loading = (el, text) => { el.replaceChildren(PL.h("div", { class: "loading", role: "status" }, text || "Loading…")); };
    PL.error = (el, e, retry) => {
        el.replaceChildren(PL.h("div", { class: "empty", role: "alert" }, e.message + " ", retry ? PL.h("button", { class: "btn btn--small btn--ghost", onclick: retry }, "Try again") : null));
    };

    // ---- router ----
    const ROUTES = ["discover", "library", "assistant"];
    PL.route = function () {
        const name = (location.hash.replace(/^#\/?/, "").split("?")[0]) || "discover";
        const route = ROUTES.includes(name) ? name : "discover";
        document.querySelectorAll(".view").forEach((v) => { v.hidden = v.dataset.view !== route; });
        document.querySelectorAll(".header__link").forEach((a) => { if (a.dataset.route === route) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current"); });
        const v = PL.views[route];
        if (v) {
            const el = document.getElementById("view-" + route);
            if (!el.dataset.ready) { el.dataset.ready = "1"; v.init(el); }
            if (v.show) v.show(el);
        }
        document.title = "PaperLens — " + route.charAt(0).toUpperCase() + route.slice(1);
        window.scrollTo(0, 0);
    };

    PL.start = function () {
        window.addEventListener("hashchange", PL.route);
        document.addEventListener("keydown", (e) => { if (e.key === "Escape") PL.closeDrawer(); });
        PL.route();
    };

    PL.openDrawer = (node) => { const d = document.getElementById("drawer"); d.replaceChildren(node); d.hidden = false; d.scrollTop = 0; const c = d.querySelector(".drawer__close"); if (c) c.focus(); };
    PL.closeDrawer = () => { document.getElementById("drawer").hidden = true; };

    /* Ask the assistant something (optionally about one paper) from anywhere in the app. */
    PL.ask = function (question, paper) {
        if (paper) PL.state.selectedDocs = [{ id: paper.id, title: paper.title }];
        PL.state.pendingQuestion = question || null;
        PL.closeDrawer();
        if (location.hash === "#/assistant") PL.route(); else location.hash = "#/assistant";
    };
    return PL;
})();
