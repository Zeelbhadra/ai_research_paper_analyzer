/* Minimal SVG/HTML charts following the data-viz rules: thin rounded bars on a baseline, a 2px gap between bars, a legend for
   >= 2 series, selective direct labels (never a number on every bar), recessive grid, hover + keyboard tooltips, and a table
   view of the same data. No dual axes. Text uses ink tokens, never series colours. */
(function () {
    "use strict";
    const { h, esc } = PL;
    const NS = "http://www.w3.org/2000/svg";
    const s = (tag, attrs) => { const e = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v); return e; };

    function niceMax(v) { if (v <= 0) return 1; const p = Math.pow(10, Math.floor(Math.log10(v))); const m = v / p; return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 5 ? 5 : 10) * p; }

    function table(columns, rows, opts) {
        const numCls = (i) => (i && !(opts && opts.text) ? "num" : "");
        return h("div", { class: "table-wrap" }, h("table", { class: "table" },
            h("thead", {}, h("tr", {}, columns.map((c, i) => h("th", { class: numCls(i) }, c)))),
            h("tbody", {}, rows.map((r) => h("tr", {}, r.map((v, i) => h("td", { class: numCls(i) }, v == null ? "n/a" : String(v))))))));
    }

    function frame(opts) {
        const wrap = h("figure", { class: "chart", style: "margin:0" });
        const head = h("div", { class: "chart__head" }, h("span", { class: "chart__title" }, opts.title));
        const body = h("div", {});
        const tbl = h("div", { hidden: true });
        let showTable = false;
        const toggle = h("button", { class: "chart__toggle", type: "button", "aria-pressed": "false", onclick: () => {
            showTable = !showTable; body.hidden = showTable; tbl.hidden = !showTable;
            toggle.textContent = showTable ? "View as chart" : "View as table"; toggle.setAttribute("aria-pressed", String(showTable));
        } }, "View as table");
        head.append(toggle);
        wrap.append(head);
        if (opts.series && opts.series.length > 1) {
            wrap.append(h("div", { class: "chart__legend" }, opts.series.map((se) => h("span", {}, h("i", { style: "background:" + se.color }), se.name))));
        }
        wrap.append(body, tbl);
        if (opts.note) wrap.append(h("figcaption", { class: "note" }, opts.note));
        return { wrap, body, tbl };
    }

    /* groupedBars({title, categories, series:[{name,color,values}], format, max, note}) */
    function groupedBars(container, o) {
        const { wrap, body, tbl } = frame(o);
        const fmt = o.format || ((v) => v.toFixed(2));
        const W = 480, H = 220, m = { l: 38, r: 8, t: 14, b: 32 };
        const all = o.series.flatMap((se) => se.values.filter((v) => v != null));
        const ymax = o.max || niceMax(Math.max(...all, 0) * 1.08);
        const iw = W - m.l - m.r, ih = H - m.t - m.b;
        const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.title });
        const y = (v) => m.t + ih - (v / ymax) * ih;
        for (let i = 0; i <= 4; i++) {
            const v = (ymax * i) / 4;
            svg.append(s("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), class: "gridline" }));
            const t = s("text", { x: m.l - 6, y: y(v) + 3, "text-anchor": "end", class: "axis" }); t.textContent = fmt(v); svg.append(t);
        }
        const gw = iw / o.categories.length, n = o.series.length, bw = Math.min(22, (gw * 0.8 - (n - 1) * 2) / n);
        o.categories.forEach((cat, ci) => {
            const gx = m.l + ci * gw + (gw - (n * bw + (n - 1) * 2)) / 2;
            let topIdx = -1, topVal = -Infinity;
            o.series.forEach((se, si) => { const v = se.values[ci]; if (v != null && v > topVal) { topVal = v; topIdx = si; } });
            o.series.forEach((se, si) => {
                const v = se.values[ci]; if (v == null) return;
                const x = gx + si * (bw + 2), top = y(v), r = Math.min(4, bw / 2, y(0) - top);
                const d = `M${x},${y(0)} V${top + r} Q${x},${top} ${x + r},${top} H${x + bw - r} Q${x + bw},${top} ${x + bw},${top + r} V${y(0)} Z`;
                const bar = s("path", { d, fill: se.color, class: "bar", tabindex: "0", "aria-label": `${se.name}, ${cat}: ${fmt(v)}` });
                const tip = `<strong>${esc(se.name)}</strong><br>${esc(cat)}: ${esc(fmt(v))}`;
                bar.addEventListener("mousemove", (e) => PL.tip.show(e, tip)); bar.addEventListener("mouseleave", PL.tip.hide);
                bar.addEventListener("focus", () => { const r2 = bar.getBoundingClientRect(); PL.tip.show({ clientX: r2.left, clientY: r2.top - 40 }, tip); }); bar.addEventListener("blur", PL.tip.hide);
                svg.append(bar);
                if (si === topIdx) { const t = s("text", { x: x + bw / 2, y: top - 4, "text-anchor": "middle", class: "val" }); t.textContent = fmt(v); svg.append(t); }
            });
            const lbl = s("text", { x: m.l + ci * gw + gw / 2, y: H - 12, "text-anchor": "middle", class: "axis" }); lbl.textContent = cat; svg.append(lbl);
        });
        svg.append(s("line", { x1: m.l, x2: W - m.r, y1: y(0), y2: y(0), stroke: "rgba(255,255,255,.25)" }));
        body.append(svg);
        tbl.append(table(["", ...o.series.map((x) => x.name)], o.categories.map((c, i) => [c, ...o.series.map((x) => x.values[i] == null ? null : fmt(x.values[i]))])));
        container.append(wrap);
    }

    /* hbars({title, items:[{label,value,color,sub}], format, max, note}) - one value per bar, all labelled (few items) */
    function hbars(container, o) {
        const { wrap, body, tbl } = frame(o);
        const fmt = o.format || ((v) => v.toFixed(2));
        const max = o.max || Math.max(...o.items.map((i) => i.value || 0), 1e-9);
        o.items.forEach((it) => {
            const row = h("div", { class: "hbar", tabindex: "0", "aria-label": `${it.label}: ${fmt(it.value)}` },
                h("span", { class: "hbar__label", title: it.label }, it.label),
                h("span", { class: "hbar__track" }, h("span", { class: "hbar__fill", style: `display:block;width:${Math.max(0.5, ((it.value || 0) / max) * 100)}%;background:${it.color || "var(--viz-tfidf)"}` })),
                h("span", { class: "hbar__val" }, fmt(it.value)));
            if (it.sub) { row.addEventListener("mousemove", (e) => PL.tip.show(e, esc(it.sub))); row.addEventListener("mouseleave", PL.tip.hide); }
            body.append(row);
        });
        tbl.append(table(["", o.valueName || "value"], o.items.map((i) => [i.label, fmt(i.value)])));
        container.append(wrap);
    }

    /* divergingBars({title, items:[{label,value,detail}], format}) - positive/negative around a zero line (two hues + neutral axis) */
    function divergingBars(container, o) {
        const { wrap, body, tbl } = frame(o);
        const fmt = o.format || ((v) => (v >= 0 ? "+" : "") + v.toFixed(2));
        const max = Math.max(...o.items.map((i) => Math.abs(i.value)), 1e-9);
        o.items.forEach((it) => {
            const w = (Math.abs(it.value) / max) * 50;
            const fill = h("span", { class: "dbar__fill", style: `background:${it.value >= 0 ? "var(--viz-pos)" : "var(--viz-neg)"};width:${w}%;${it.value >= 0 ? "left:50%" : "right:50%"}` });
            const row = h("div", { class: "dbar", tabindex: "0", "aria-label": `${it.label}: ${fmt(it.value)} ${it.value >= 0 ? "raises" : "lowers"} relevance` },
                h("span", { class: "hbar__label", title: it.label }, it.label), h("span", { class: "dbar__track" }, fill), h("span", { class: "hbar__val" }, (it.value >= 0 ? "▲ " : "▼ ") + fmt(it.value)));
            if (it.detail) { row.addEventListener("mousemove", (e) => PL.tip.show(e, esc(it.detail))); row.addEventListener("mouseleave", PL.tip.hide); }
            body.append(row);
        });
        body.append(h("div", { class: "chart__legend" }, h("span", {}, h("i", { style: "background:var(--viz-pos)" }), "▲ raises relevance"), h("span", {}, h("i", { style: "background:var(--viz-neg)" }), "▼ lowers relevance")));
        tbl.append(table(["feature", "contribution"], o.items.map((i) => [i.label, fmt(i.value)])));
        container.append(wrap);
    }

    PL.charts = { groupedBars, hbars, divergingBars, table };
})();
