(function () {
    "use strict";
    const { h, fmt } = PL;
    let root, body;

    function tile(label, value, sub, small) {
        return h("div", { class: "card stat" }, h("span", { class: "stat__label" }, label),
            h("span", { class: "stat__value" + (small ? " stat__value--sm" : "") }, value), sub ? h("span", { class: "stat__sub" }, sub) : null);
    }

    async function load() {
        try {
            const [d, bm] = await Promise.all([PL.api("/api/dashboard"), PL.api("/api/benchmarks/overview")]);
            render(d, bm);
        } catch (e) { PL.error(body, e, load); }
    }

    function render(d, bm) {
        const c = d.corpus, am = d.active_model, lb = d.latest_benchmark, rs = d.retrieval_serving, fb = d.fallback, pl = d.pipeline;
        const perf = am && bm.performance ? bm.performance[am.name] : null;
        const lat = rs && rs.latency_ms && rs.latency_ms.n ? rs.latency_ms : null;
        body.replaceChildren(
            h("div", { class: "grid grid--tiles" },
                tile("Papers indexed", fmt.int(c.papers), `${fmt.int(c.processed)} processed · ${c.failed} failed`),
                tile("Processed documents", fmt.int(c.processed), c.extraction_failures ? `${c.extraction_failures} document(s) failed extraction` : "no extraction failures"),
                tile("Chunks", fmt.int(c.chunks), "256-word chunks, 15% overlap"),
                tile("Active embedding model", am ? PL.modelLabel(am.name) : "none", am ? `promoted ${fmt.date(am.promoted_at)}` : "run a benchmark first", true),
                tile("Vector database", d.vector_db.name, d.vector_db.kind, true),
                tile("Avg retrieval latency", lat ? fmt.ms(lat.mean) : (perf ? fmt.ms(perf.latency_mean_ms) : "n/a"),
                    lat ? `live, last 24 h · p95 ${fmt.ms(lat.p95)} · ${lat.n} requests` : (perf ? `benchmarked (isolated, CPU) · p95 ${fmt.ms(perf.latency_p95_ms)}; no live traffic recorded yet` : "no data"))),
            h("div", { class: "spacer" }),
            h("div", { class: "grid grid--2" },
                h("div", { class: "card" }, h("h3", {}, "Retrieval status"), statusCard(fb, am)),
                h("div", { class: "card", id: "dash-bench" }, h("h3", {}, "Latest benchmark", lb ? h("small", {}, `${lb.run_id} · ${lb.split} split · ${lb.n_queries} queries · ${fmt.date(lb.timestamp)}`) : null))),
            h("div", { class: "spacer" }),
            h("div", { class: "grid grid--2" },
                h("div", { class: "card" }, h("h3", {}, "Pipeline"), pipelineCard(pl)),
                h("div", { class: "card" }, h("h3", {}, "Data caveats"),
                    h("p", { class: "small muted" }, "Benchmark queries are synthetic and not human-verified; quality numbers are therefore indicative. See the Benchmarks view for the limitations of each measurement."))));
        const mount = document.getElementById("dash-bench");
        if (lb) {
            PL.charts.hbars(mount, {
                title: "nDCG@10 by model (higher is better)", valueName: "nDCG@10", max: 1, format: (v) => v.toFixed(3),
                items: lb.candidates.map((x) => ({ label: PL.modelLabel(x.name) + (x.name === lb.winner ? "  ★ selected" : ""), value: x["nDCG@10"], color: PL.modelColor(x.name) })).sort((a, b) => b.value - a.value),
                note: lb.winner ? `Selected: ${PL.modelLabel(lb.winner)} (highest nDCG@10 among feasible models unless a cheaper one is demonstrably not worse).` : "",
            });
        } else mount.append(h("div", { class: "empty" }, "No benchmark has been run yet."));
    }

    function statusCard(fb, am) {
        if (!fb) return h("div", {}, h("span", { class: "chip" }, "not loaded yet"), h("p", { class: "small muted" }, "The retrieval service loads on first use (Assistant, search or explainability). Status appears here afterwards."),
            h("button", { class: "btn btn--small btn--ghost", onclick: async (e) => { e.target.disabled = true; e.target.textContent = "Loading models…"; try { await PL.api("/api/retrieval/status"); } catch (x) { PL.toast(x.message, "error"); } load(); } }, "Load now"));
        const active = fb.fallback_active;
        return h("div", {},
            h("div", { class: "row" }, h("span", { class: "chip " + (active ? "chip--warn" : "chip--ok") }, active ? "⚠ FALLBACK ACTIVE" : "✔ primary healthy"), h("span", { class: "muted small" }, `serving: ${PL.modelLabel(active ? fb.fallback : fb.primary)}`)),
            h("dl", { class: "kv", style: "margin-top:.6rem" },
                h("dt", {}, "Primary"), h("dd", {}, PL.modelLabel(fb.primary)), h("dt", {}, "Fallback"), h("dd", {}, fb.fallback ? PL.modelLabel(fb.fallback) : "none (production is the lexical model)"),
                h("dt", {}, "Requests / via fallback"), h("dd", {}, `${fb.total_requests} / ${fb.fallback_requests}`), h("dt", {}, "Primary failures"), h("dd", {}, String(fb.total_failures)),
                active ? h("dt", {}, "Reason") : null, active ? h("dd", {}, PL.fmt.short(fb.reason, 160)) : null),
            h("div", { class: "row", style: "margin-top:.6rem" }, h("button", { class: "btn btn--small btn--ghost", onclick: async () => { try { const r = await PL.api("/api/retrieval/health-check?force=1", { method: "POST" }); PL.toast(`Health check: ${r.ok ? "ok" : "failed"} (${r.detail})`, r.ok ? "success" : "error"); } catch (x) { PL.toast(x.message, "error"); } load(); } }, "Run health check")));
    }

    function pipelineCard(pl) {
        if (!pl) return h("div", { class: "muted" }, "n/a");
        const by = pl.jobs_by_status || {};
        return h("div", {},
            h("div", { class: "row" }, ["success", "failed", "running", "partial"].filter((k) => by[k]).map((k) => h("span", { class: "chip " + ({ success: "chip--ok", failed: "chip--bad", partial: "chip--warn" }[k] || "") }, `${k}: ${by[k]}`)),
                Object.keys(by).length ? null : h("span", { class: "muted" }, "no jobs recorded yet")),
            pl.running_workflows && pl.running_workflows.length ? h("p", { class: "small" }, "Running now: " + pl.running_workflows.join(", ")) : null,
            (pl.recent_failures || []).map((j) => h("p", { class: "small muted" }, `✖ ${j.kind} (${fmt.date(j.started)})`)),
            h("a", { href: "#/pipeline", class: "small" }, "Open pipeline monitoring →"));
    }

    PL.views.dashboard = {
        refreshEvery: 20000,
        init(el) {
            root = el;
            el.append(h("h2", { class: "view__title" }, "Dashboard"), h("p", { class: "view__lead" }, "Corpus, active retrieval model, serving health and the latest benchmark at a glance."));
            body = h("div", {}); el.append(body); PL.loading(body);
        },
        show() { load(); },
    };
})();
