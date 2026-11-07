(function () {
    "use strict";
    const { h, fmt } = PL;
    let body;

    const ICON = { success: ["✔", "chip--ok"], failed: ["✖", "chip--bad"], running: ["●", "chip--accent"], partial: ["◐", "chip--warn"], skipped: ["—", ""] };
    const badge = (st) => { const [i, c] = ICON[st] || ["?", ""]; return h("span", { class: "chip " + c }, `${i} ${st}`); };
    const dur = (j) => (j.finished && j.started ? fmt.secs(j.finished - j.started) : j.status === "running" ? "running…" : "n/a");
    const detail = (j) => { try { const d = JSON.parse(j.detail || "{}"); return PL.fmt.short(d.error || (d.attempts ? `attempts: ${d.attempts}` : Object.entries(d).map(([k, v]) => `${k}: ${v}`).join(", ")), 80); } catch (_) { return PL.fmt.short(j.detail || "", 80); } };

    async function load() {
        try {
            const [st, jobs, fails, bm, ev] = await Promise.all([PL.api("/api/pipeline/status"), PL.api("/api/pipeline/jobs?limit=60"), PL.api("/api/pipeline/failures"),
                PL.api("/api/benchmarks/overview").catch(() => null), PL.api("/api/retrieval/events").catch(() => [])]);
            render(st, jobs, fails, bm, ev);
        } catch (e) { PL.error(body, e, load); }
    }

    function render(st, jobs, fails, bm, events) {
        const counts = {}; jobs.forEach((j) => { counts[j.status] = (counts[j.status] || 0) + 1; });
        const wf = jobs.filter((j) => j.kind.startsWith("workflow:"));
        const topic = h("input", { class: "input", placeholder: "topic for discovery, e.g. graph neural networks", "aria-label": "Discovery topic", style: "min-width:260px" });
        const sel = h("select", { class: "input", "aria-label": "Workflow" }, [["ingest_topic", "Discover and ingest papers on a topic"], ["monitor", "Run monitoring check (benchmark only if a rule fires)"], ["benchmark_and_deploy", "Run benchmark and deploy validated winner"], ["report", "Generate report"]].map(([v, n]) => h("option", { value: v }, n)));
        const run = h("button", { class: "btn btn--primary", onclick: async () => {
            const params = sel.value === "ingest_topic" ? { topic: topic.value.trim() } : {};
            if (sel.value === "ingest_topic" && !params.topic) return PL.toast("Enter a topic first.", "error");
            run.disabled = true;
            try { await PL.api("/api/pipeline/run", { json: { workflow: sel.value, params } }); PL.toast("Started " + sel.value, "success"); setTimeout(load, 800); }
            catch (e) { PL.toast(e.status === 409 ? "That workflow is already running." : e.message, "error"); } finally { run.disabled = false; }
        } }, "Run now");
        body.replaceChildren(
            h("div", { class: "grid grid--tiles" },
                h("div", { class: "card stat" }, h("span", { class: "stat__label" }, "Running now"), h("span", { class: "stat__value" }, String(st.running.length)), h("span", { class: "stat__sub" }, st.running.join(", ") || "idle")),
                h("div", { class: "card stat" }, h("span", { class: "stat__label" }, "Scheduler"), h("span", { class: "stat__value stat__value--sm" }, st.scheduler.running ? "running" : st.scheduler.enabled ? "enabled (not started)" : "off"), h("span", { class: "stat__sub" }, st.scheduler.running ? st.scheduler.jobs.length + " scheduled jobs" : "set AUTOMATION=1 to enable")),
                h("div", { class: "card stat" }, h("span", { class: "stat__label" }, "Failed documents"), h("span", { class: "stat__value" }, String(fails.length)), h("span", { class: "stat__sub" }, fails.length ? "see list below" : "none")),
                h("div", { class: "card stat" }, h("span", { class: "stat__label" }, "Recent jobs"), h("div", { class: "row" }, Object.entries(counts).map(([k, v]) => h("span", { class: "chip " + (ICON[k] || ["", ""])[1] }, `${k} ${v}`)), Object.keys(counts).length ? null : h("span", { class: "muted" }, "none")))),
            h("div", { class: "spacer" }),
            h("div", { class: "card" }, h("h3", {}, "Run a workflow"), h("div", { class: "row" }, sel, topic, run), h("p", { class: "note" }, "Same code path as the scheduler. A workflow that is already running is not started twice.")),
            h("div", { class: "spacer" }),
            h("div", { class: "grid grid--2" },
                h("div", { class: "card" }, h("h3", {}, "Ingestion, processing, embedding and benchmark jobs"), h("div", { class: "table-wrap", style: "max-height:420px;overflow:auto" }, jobTable(jobs))),
                h("div", { class: "card" }, h("h3", {}, "Failed documents"), fails.length ? h("div", { class: "table-wrap" }, PL.charts.table(["Document", "Reason", "When"], fails.map((f) => [f.path.split(/[\\/]/).pop(), PL.fmt.short(f.reason, 90), fmt.date(f.ts)]), { text: true })) : h("span", { class: "muted" }, "No extraction failures."),
                    h("p", { class: "note" }, "Corrupt, truncated or scanned PDFs are logged here and never block the rest of a batch. Scanned PDFs are detected but not OCR'd."))),
            h("div", { class: "spacer" }),
            h("div", { class: "grid grid--2" },
                h("div", { class: "card" }, h("h3", {}, "Model switching history"), bm && bm.promotions.length ? PL.charts.table(["When", "From", "To", "Run"], bm.promotions.map((p) => [fmt.date(p.at), p.from ? PL.modelLabel(p.from) : "—", PL.modelLabel(p.to), p.run_id]), { text: true }) : h("span", { class: "muted" }, "none yet"),
                    h("p", { class: "note" }, "A model is promoted only after it passes feasibility checks and a live validation probe; otherwise production is left untouched.")),
                h("div", { class: "card" }, h("h3", {}, "Retrieval service events"), events.length ? h("div", { class: "table-wrap", style: "max-height:300px;overflow:auto" }, PL.charts.table(["Time", "Event", "Detail"], events.slice().reverse().slice(0, 30).map((e) => [e.ts, e.kind, PL.fmt.short(e.reason || e.message || e.error || (e.new ? e.old + " → " + e.new : "") || "", 70)]), { text: true })) : h("span", { class: "muted" }, "No events (service not loaded or no incidents)."))),
            h("div", { class: "spacer" }), reportCard());
    }

    function jobTable(jobs) {
        return h("table", { class: "table" }, h("thead", {}, h("tr", {}, ["#", "Job", "Status", "Started", "Took", "Detail"].map((c) => h("th", {}, c)))),
            h("tbody", {}, jobs.map((j) => h("tr", {}, h("td", {}, j.id), h("td", {}, j.kind), h("td", {}, badge(j.status)), h("td", {}, fmt.date(j.started)), h("td", {}, dur(j)), h("td", { class: "small muted" }, detail(j))))));
    }

    function reportCard() {
        const box = h("div", { class: "card" }, h("h3", {}, "Monitoring report"));
        PL.api("/api/pipeline/report").then((r) => {
            const c = r.check;
            box.append(h("p", { class: "small muted" }, "Latest generated report: " + String(r.markdown).split(/[\\/]/).pop()),
                c.alerts.length ? c.alerts.map((a) => h("div", { class: "banner", style: "margin:.3rem 0" }, h("strong", {}, a.type.replace(/_/g, " ")), " — " + a.detail)) : h("span", { class: "chip chip--ok" }, "✔ no alerts"),
                h("p", { class: "small" }, c.trigger_benchmark ? "Benchmark re-run recommended: " + c.benchmark_trigger_reasons.join("; ") : "No benchmark re-run needed (no trigger rule fired)."),
                h("p", { class: "small muted" }, `Window: ${c.window_minutes} min · ${c.requests} requests · fallback rate ${c.fallback_rate == null ? "n/a" : fmt.pct(c.fallback_rate)}`));
        }).catch(() => box.append(h("span", { class: "muted" }, "No report yet. Use “Run now” → Generate report."), ));
        return box;
    }

    PL.views.pipeline = {
        refreshEvery: 8000,
        init(el) { el.append(h("h2", { class: "view__title" }, "Pipeline monitoring"), h("p", { class: "view__lead" }, "Ingestion, processing, indexing and benchmark jobs, failures, model switches and monitoring alerts."), (body = h("div", {}))); PL.loading(body); },
        show() { load(); },
    };
})();
