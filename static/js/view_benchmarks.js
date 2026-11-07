(function () {
    "use strict";
    const { h, fmt } = PL;
    const ORDER = ["tfidf", "sbert", "e5", "bge", "hybrid"];
    const keyOf = (n) => (n.startsWith("hybrid") ? "hybrid" : n);
    let body, data, runId = null, split = "validation";

    async function load() {
        PL.loading(body);
        try { data = await PL.api("/api/benchmarks/overview"); if (!runId) { const real = data.runs.filter((r) => !r.adjudicated); runId = (real[real.length - 1] || data.runs[data.runs.length - 1] || {}).run_id; } render(); }
        catch (e) { PL.error(body, e, load); }
    }

    const sorted = (names) => names.slice().sort((a, b) => ORDER.indexOf(keyOf(a)) - ORDER.indexOf(keyOf(b)));
    const series = (names, fn) => sorted(names).map((n) => ({ name: PL.modelLabel(n), color: PL.modelColor(n), values: fn(n) }));

    function render() {
        body.replaceChildren();
        if (!data.runs.length) { body.append(h("div", { class: "empty" }, "No benchmark has been run yet. Run `python -m backend.benchmarking.run_benchmark`.")); return; }
        const run = data.runs.find((r) => r.run_id === runId) || data.runs[data.runs.length - 1];
        const prod = data.production;
        const names = run.candidates.map((c) => c.name);
        const eligible = run.candidates.filter((c) => c.eligible);

        // ---- header: production / winner / fallback ----
        body.append(h("div", { class: "grid grid--3" },
            h("div", { class: "card stat" }, h("span", { class: "stat__label" }, "Current production model"), h("span", { class: "stat__value stat__value--sm" }, prod ? PL.modelLabel(prod.name) : "none"), h("span", { class: "stat__sub" }, prod ? `promoted ${fmt.date(prod.promoted_at)} (${prod.run_id})` : "")),
            h("div", { class: "card stat" }, h("span", { class: "stat__label" }, "Highest-scoring validated model in this run"), h("span", { class: "stat__value stat__value--sm" }, PL.modelLabel(run.winner || run.reference)),
                h("span", { class: "stat__sub" }, `selected on the ${run.split} split (${run.n_queries} queries); ${eligible.length}/${run.candidates.length} candidates feasible`)),
            h("div", { class: "card stat" }, h("span", { class: "stat__label" }, "TF-IDF fallback"), h("span", { class: "stat__value stat__value--sm" }, prod && prod.family === "tfidf" ? "same as production (no independent fallback)" : "standby (independent lexical index)"),
                h("a", { class: "stat__sub", href: "#/dashboard" }, "live fallback status on the dashboard →"))));

        // ---- controls ----
        const runSel = h("select", { class: "input", "aria-label": "Benchmark run", onchange: (e) => { runId = e.target.value; render(); } },
            data.runs.map((r) => h("option", { value: r.run_id, selected: r.run_id === run.run_id || null }, `${r.run_id} · ${fmt.date(r.timestamp)} · ${r.split}${r.adjudicated ? " · adjudicated labels" : ""}`)));
        const splitSel = h("select", { class: "input", "aria-label": "Evaluation split", onchange: (e) => { split = e.target.value; render(); } },
            ["validation", "test", "dev"].map((s) => h("option", { value: s, selected: s === split || null }, s + (s === "test" ? " (held-out, reported only)" : s === "dev" ? " (used to tune hybrid weights)" : " (used for selection)"))));
        body.append(h("div", { class: "row", style: "margin:1rem 0" }, h("label", { class: "field" }, "Benchmark run", runSel), h("label", { class: "field" }, "Ranking-metric split", splitSel),
            h("span", { class: "muted small" }, "Per-K ranking charts use the stored per-query results of the latest evaluation of the chosen split.")));

        // ---- ranking quality ----
        const rm = (data.ranking_metrics || {})[split];
        const quality = h("div", { class: "grid grid--charts" });
        if (rm) {
            const mn = sorted(Object.keys(rm)), Ks = [1, 3, 5, 10];
            [["Precision@K", "P"], ["Recall@K", "R"], ["nDCG@K", "nDCG"]].forEach(([title, p]) => {
                const box = h("div", { class: "card" }); quality.append(box);
                PL.charts.groupedBars(box, { title: `${title} (${split}, n=${rm[mn[0]].n})`, categories: Ks.map((k) => "K=" + k), format: (v) => v.toFixed(2),
                    series: series(mn, (n) => Ks.map((k) => rm[n][`${p}@${k}`])), max: p === "P" ? undefined : 1 });
            });
            const box = h("div", { class: "card" }); quality.append(box);
            PL.charts.groupedBars(box, { title: `MRR@10 and MAP (${split})`, categories: ["MRR@10", "MAP"], format: (v) => v.toFixed(2), max: 1, series: series(mn, (n) => [rm[n]["MRR@10"], rm[n].MAP]) });
        } else quality.append(h("div", { class: "empty" }, `No stored per-query results for the ${split} split.`));
        body.append(h("h3", { style: "margin:1.2rem 0 .5rem" }, "Retrieval quality"), quality,
            h("p", { class: "note" }, "ROC-AUC / PR-AUC / F1 / accuracy are not shown for the retrieval models: they produce rankings, not binary relevance decisions, so those metrics would not be meaningful. They are reported for the relevance classifier in the Explainability view. Queries are synthetic and not human-verified (see Dataset below); each chart has a table view."));

        // ---- efficiency ----
        const perf = data.performance;
        if (perf) {
            const pn = sorted(Object.keys(perf)), grid = h("div", { class: "grid grid--charts" }); body.append(h("h3", { style: "margin:1.2rem 0 .5rem" }, "Efficiency (isolated runs, CPU, one process per model)"), grid);
            const mk = (title, fn, f, note, vn) => { const b = h("div", { class: "card" }); grid.append(b); PL.charts.hbars(b, { title, valueName: vn, items: pn.map((n) => ({ label: PL.modelLabel(n), value: fn(perf[n]), color: PL.modelColor(n) })), format: f, note }); };
            const lb = h("div", { class: "card" }); grid.append(lb);
            PL.charts.groupedBars(lb, { title: "Retrieval latency (lower is better)", categories: ["p50", "p95"], format: (v) => v.toFixed(0) + " ms", series: series(pn, (n) => [perf[n].latency_p50_ms, perf[n].latency_p95_ms]) });
            mk("Throughput (queries/s, higher is better)", (p) => p.throughput_qps, (v) => v.toFixed(0), "Sequential single client.", "qps");
            mk("Memory growth after load (lower is better)", (p) => p.memory_mb, (v) => fmt.mb(v), "RSS increase of a fresh process.", "MB");
            mk("Indexing time (lower is better)", (p) => p.indexing_seconds, (v) => fmt.secs(v), "Dense models: extrapolated from a 192-chunk sample. TF-IDF measured.", "seconds");
            mk("Index size on disk", (p) => p.index_bytes / 1048576, (v) => v.toFixed(1) + " MB", "", "MB");
        }

        // ---- selection ----
        body.append(h("h3", { style: "margin:1.2rem 0 .5rem" }, `Selection — ${run.run_id}`),
            h("div", {}, h("div", { class: "card" }, h("h3", {}, "How the model was chosen"), h("ol", { class: "small", style: "padding-left:1.2rem" }, (run.trace || []).map((t) => h("li", {}, t))),
                h("p", { class: "note" }, "No hand-set metric weights. Feasibility (quality floor, health, resource budgets) → highest nDCG@10 → a cheaper model replaces it only if a paired-bootstrap interval shows it is not materially worse → smallest footprint breaks ties.")),
                h("div", { class: "spacer" }), h("div", { class: "card" }, h("h3", {}, "Candidates"), candidateTable(run, prod))));

        // ---- held-out ----
        if (run.heldout_test) {
            const ht = run.heldout_test, tn = sorted(Object.keys(ht));
            body.append(h("div", { class: "card", style: "margin-top:1rem" }, h("h3", {}, "Held-out test split ", h("small", {}, "reported after selection; never used to choose")),
                PL.charts.table(["Model", "nDCG@10", "95% CI", "Recall@5", "MRR@10", "p vs best"], tn.map((n) => [PL.modelLabel(n), fmt.n(ht[n]["nDCG@10"]), `${fmt.n(ht[n]["nDCG@10_ci95"][0], 2)}–${fmt.n(ht[n]["nDCG@10_ci95"][1], 2)}`, fmt.n(ht[n]["R@5"]), fmt.n(ht[n]["MRR@10"]), ht[n]["p_vs_best_nDCG@10"] == null ? "best" : fmt.n(ht[n]["p_vs_best_nDCG@10"])]))));
        }

        // ---- by category ----
        if (data.by_category) {
            const cats = Object.keys(Object.values(data.by_category)[0].by_category), bn = sorted(Object.keys(data.by_category));
            body.append(h("div", { class: "card", style: "margin-top:1rem" }, h("h3", {}, "nDCG@10 by query category ", h("small", {}, "validation split; each cell has only 3–4 queries")),
                PL.charts.table(["Category", ...bn.map(PL.modelLabel)], cats.map((c) => [c.replace(/_/g, " "), ...bn.map((n) => fmt.n(data.by_category[n].by_category[c]["nDCG@10"], 2))]))));
        }

        // ---- supporting studies ----
        body.append(h("h3", { style: "margin:1.2rem 0 .5rem" }, "Supporting studies"), studies());

        // ---- history ----
        body.append(h("h3", { style: "margin:1.2rem 0 .5rem" }, "History"),
            h("div", { class: "grid grid--2" }, h("div", { class: "card" }, h("h3", {}, "All benchmark runs"),
                PL.charts.table(["Run", "Date", "Split", "Queries", "Winner"], data.runs.map((r) => [r.run_id + (r.adjudicated ? " (adj.)" : ""), fmt.date(r.timestamp), r.split, r.n_queries, PL.modelLabel(r.winner)]), { text: true })),
                h("div", { class: "card" }, h("h3", {}, "Model switching history"), data.promotions.length ? PL.charts.table(["When", "From", "To", "Run", "Reason"], data.promotions.map((p) => [fmt.date(p.at), p.from ? PL.modelLabel(p.from) : "—", PL.modelLabel(p.to), p.run_id, PL.fmt.short(p.reason, 60)]), { text: true }) : h("span", { class: "muted" }, "none"))));
    }

    function candidateTable(run, prod) {
        const rows = run.candidates.slice().sort((a, b) => b.metrics["nDCG@10"] - a.metrics["nDCG@10"]).map((c) => {
            const m = c.metrics, ni = c.non_inferiority;
            return [PL.modelLabel(c.name) + (prod && prod.name === c.name ? " ★" : ""), fmt.n(m["nDCG@10"]), fmt.n(m["Recall@5"]), fmt.n(m["MRR@10"]), fmt.n(m["Precision@5"]), fmt.n(m.MAP), fmt.n(c.quality_index, 1),
                fmt.ms(m.latency_p50_ms), fmt.mb(m.memory_mb), fmt.secs(m.indexing_seconds), c.footprint != null ? fmt.n(c.footprint, 1) : "n/a",
                ni ? `${ni.mean_diff >= 0 ? "+" : ""}${fmt.n(ni.mean_diff)} [${fmt.n(ni.ci_lo, 2)}, ${fmt.n(ni.ci_hi, 2)}]${ni.non_inferior ? " ✔" : ""}` : "reference",
                (c.eligible ? "✔ feasible" : "✖ " + c.fail_reasons.join("; ")) + (c.pareto_optimal ? " · Pareto" : "")];
        });
        return PL.charts.table(["Model", "nDCG@10", "R@5", "MRR@10", "P@5", "MAP", "Quality idx", "p50", "Memory", "Index time", "Footprint", "Δ nDCG vs reference [95% CI]", "Status"], rows);
    }

    function studies() {
        const g = h("div", { class: "grid grid--charts" });
        const sw = data.chunk_sweep && data.chunk_sweep.validation;
        if (sw) {
            const keys = Object.keys(sw.results), c = h("div", { class: "card" }); g.append(c);
            PL.charts.groupedBars(c, { title: `Chunking sweep: evidence covered within a ${sw.budget_words || 1500}-word context (validation)`, categories: keys.map((k) => k.replace("recursive-", "rec ").replace("-15", "").replace("-", "/")), format: (v) => v.toFixed(2), max: 1,
                series: [{ name: "TF-IDF", color: "var(--viz-tfidf)", values: keys.map((k) => sw.results[k].tfidf.ctx_coverage) }, { name: "SBERT", color: "var(--viz-sbert)", values: keys.map((k) => sw.results[k].sbert.ctx_coverage) }],
                note: "Fixed context budget, so chunk size cannot win just by being bigger. 256 words / 15% overlap (recursive) is best on dev and validation. Only TF-IDF and SBERT were swept; wide confidence intervals (see JSON)." });
        }
        const rr = data.reranker;
        if (rr) g.append(h("div", { class: "card" }, h("h3", {}, "Cross-encoder reranker ", h("span", { class: "chip " + (rr.adopt ? "chip--ok" : "chip--warn") }, rr.adopt ? "adopted" : "not adopted")),
            PL.charts.table(["Split", "n", "Baseline nDCG@10", "Reranked", "Δ [95% CI]"], Object.entries(rr.splits).map(([s, v]) => [s, v.n, fmt.n(v.baseline_nDCG10 != null ? v.baseline_nDCG10 : v["baseline_nDCG@10"]), fmt.n(v["reranked_nDCG@10"]), `${v.diff >= 0 ? "+" : ""}${fmt.n(v.diff)} [${fmt.n(v.diff_ci95[0], 2)}, ${fmt.n(v.diff_ci95[1], 2)}]`])),
            h("p", { class: "note" }, `Rule fixed in advance: adopt only if the validation CI lower bound is above 0. Cost: about ${rr.rerank_seconds_per_query_measured || "?"} s/query on CPU.`)));
        const pp = data.paraphrase_probe;
        if (pp) { const c = h("div", { class: "card" }); g.append(c);
            PL.charts.groupedBars(c, { title: "Lexical-bias probe: nDCG@10 on original vs reworded queries", categories: ["original", "reworded"], format: (v) => v.toFixed(2), max: 1,
                series: sorted(Object.keys(pp.models)).map((n) => ({ name: PL.modelLabel(n), color: PL.modelColor(n), values: [pp.models[n]["nDCG@10_original"], pp.models[n]["nDCG@10_paraphrase"]] })),
                note: `Rewording cut query–passage word overlap from ${fmt.n(pp.mean_overlap_before, 2)} to ${fmt.n(pp.mean_overlap_after, 2)}. TF-IDF drops most, so the original queries favour lexical matching.` }); }
        const ds = data.dataset;
        if (ds) g.append(h("div", { class: "card" }, h("h3", {}, "Benchmark dataset ", h("small", {}, "v" + ds.version)),
            h("dl", { class: "kv" }, h("dt", {}, "Queries"), h("dd", {}, `${ds.retained} retained of ${ds.generated} generated`), h("dt", {}, "Splits (by paper)"), h("dd", {}, Object.entries(ds.splits).map(([k, v]) => `${k} ${v}`).join(" · ")),
                h("dt", {}, "Categories"), h("dd", {}, Object.entries(ds.categories).map(([k, v]) => `${k.replace(/_/g, " ")} ${v}`).join(" · ")), h("dt", {}, "Labels"), h("dd", {}, `synthetic (LLM-generated, auto-validated); human-verified: ${ds.human_verified}`)),
            h("p", { class: "note" }, "Labels are incomplete (each query is labelled from one passage) and the queries share vocabulary with their source passage. Treat absolute numbers as indicative; export storage/benchmark/review.csv to review queries.")));
        return g;
    }

    PL.views.benchmarks = {
        init(el) { el.append(h("h2", { class: "view__title" }, "Model benchmarking"), h("p", { class: "view__lead" }, "TF-IDF, SBERT, E5, BGE and the TF-IDF+E5 hybrid, measured on identical queries and chunks. Nothing here is estimated unless marked."), (body = h("div", {}))); },
        show() { if (!data) load(); },
    };
})();
