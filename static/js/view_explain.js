(function () {
    "use strict";
    const { h, fmt } = PL;
    let retOut, ansOut, modelOut;

    // ---------- retrieval explanation ----------
    async function explainRetrieval(q, k, lime, btn) {
        PL.loading(retOut, lime ? "Explaining (LIME takes a few seconds per result)…" : "Explaining…"); btn.disabled = true;
        try {
            const d = await PL.api(`/api/explain/retrieval?q=${encodeURIComponent(q)}&k=${k}&lime=${lime ? 1 : 0}`);
            retOut.replaceChildren(
                d.fallback_active ? h("div", { class: "banner" }, "⚠ Fallback retrieval is active; results come from TF-IDF only.") : null,
                d.model_warning ? h("div", { class: "banner banner--info" }, d.model_warning) : null,
                h("div", { class: "row small muted" }, `Retriever: ${PL.modelLabel(d.retriever)} (${d.retriever_kind})`),
                ...d.results.map(resultCard), d.caveat ? h("p", { class: "note" }, d.caveat) : null);
        } catch (e) { PL.error(retOut, e); } finally { btn.disabled = false; }
    }

    function resultCard(r) {
        const c = r.components, card = h("div", { class: "card result-ex" });
        card.append(h("div", { class: "row row--between" }, h("h3", { style: "margin:0" }, `#${r.rank} ${r.title}`), h("span", { class: "chip" }, `${r.pages[0] === r.pages[1] ? "p. " + r.pages[0] : "pp. " + r.pages[0] + "–" + r.pages[1]} · ${r.section.replace(/_/g, " ")}`)),
            h("div", { class: "row small muted" }, h("span", { class: "chip chip--accent" }, "path: " + r.retrieval_path), `fused score ${r.score}`));
        const g = h("div", { class: "grid grid--2", style: "margin-top:.6rem" });
        const left = h("div", {}), right = h("div", {}); g.append(left, right);
        // exact facts
        left.append(h("h4", { class: "small muted" }, "EXACT RETRIEVAL FACTS"), h("dl", { class: "kv" },
            h("dt", {}, "TF-IDF"), h("dd", {}, `cosine ${c.tfidf.cosine}, rank ${c.tfidf.rank_in_corpus} of corpus`), h("dt", {}, "Dense"), h("dd", {}, `cosine ${c.dense.cosine}, rank ${c.dense.rank_in_corpus} of corpus`)));
        if (r.rrf) PL.charts.hbars(left, { title: "Rank-fusion contributions", valueName: "RRF part", format: (v) => v.toFixed(4), note: r.rrf.formula,
            items: [{ label: "TF-IDF rank " + c.tfidf.rank_in_corpus, value: r.rrf.sparse_part, color: "var(--viz-tfidf)" }, { label: "Dense rank " + c.dense.rank_in_corpus, value: r.rrf.dense_part, color: "var(--viz-e5)" }] });
        if (r.tfidf_term_contributions.length) PL.charts.hbars(left, { title: "Terms driving the lexical score (exact)", valueName: "contribution", format: (v) => v.toFixed(3),
            items: r.tfidf_term_contributions.slice(0, 6).map((t) => ({ label: t.term, value: t.contribution, color: "var(--viz-tfidf)" })) });
        left.append(h("div", { class: "tagrow", style: "margin-top:.5rem" }, h("span", { class: "small muted" }, "matched query terms:"), r.matched_query_terms.map((t) => h("span", { class: "chip" }, t))));
        if (r.best_passage) left.append(h("p", { class: "small" }, h("strong", {}, "Best-matching passage "), `(dense cosine ${r.best_passage.cosine}): `, h("mark", { class: "hl" }, r.best_passage.text)));
        // model-based
        const m = r.relevance_model;
        if (m) {
            right.append(h("h4", { class: "small muted" }, "RELEVANCE MODEL (surrogate over readable features)"),
                h("div", { class: "row" }, h("strong", {}, `P(relevant) = ${m.probability.toFixed(3)}`), h("span", { class: "chip " + (m.predicted_relevant ? "chip--ok" : "chip--warn") }, (m.predicted_relevant ? "✔ predicted relevant" : "✖ predicted not relevant") + ` (threshold ${m.threshold})`)),
                h("div", { class: "gauge", role: "img", "aria-label": "probability " + m.probability }, h("i", { style: `width:${m.probability * 100}%` })));
            PL.charts.divergingBars(right, { title: "SHAP: what pushed this prediction (log-odds)", format: (v) => (v >= 0 ? "+" : "") + v.toFixed(2),
                items: m.shap.contributions.map((x) => ({ label: x.feature.replace(/_/g, " "), value: x.shap, detail: `feature value ${x.value}` })),
                note: `base ${m.shap.base_value} + contributions = ${m.shap.prediction_logodds} (additivity error ${m.shap.additivity_error}). Signs describe this surrogate model, not causal truth: e.g. title overlap has a negative weight because most hard negatives in its training pool come from the same paper as the relevant passage.` });
            if (m.lime) PL.charts.divergingBars(right, { title: "LIME: local weights", format: (v) => (v >= 0 ? "+" : "") + v.toFixed(3),
                items: m.lime.weights.map((x) => ({ label: x.condition, value: x.weight })), note: `Local linear fit R² = ${m.lime.local_model_r2}${m.lime.local_model_r2 < 0.4 ? " — low, so treat this explanation as approximate" : ""}.` });
        } else right.append(h("p", { class: "muted small" }, "Relevance-model explanation unavailable (see notice above)."));
        card.append(g);
        card.append(h("details", { class: "sec" }, h("summary", {}, "Passage text"), h("div", { class: "sec__body" }, h("div", { class: "chunk" }, r.text))));
        return card;
    }

    // ---------- relevance model summary ----------
    async function loadModel() {
        try {
            const d = await PL.api("/api/explain/model"), rep = d.report || {};
            modelOut.replaceChildren();
            if (!rep.splits) { modelOut.append(h("div", { class: "empty" }, d.warning || "Relevance model not trained.")); return; }
            const rows = ["dev", "validation", "test"].filter((s) => rep.splits[s]).map((s) => { const c = rep.splits[s].chosen_model; return [s + (s === "dev" ? " (train)" : ""), fmt.pct(rep.splits[s].pos_rate, 1), fmt.n(c.roc_auc), fmt.n(c.pr_auc), fmt.n(c.precision), fmt.n(c.recall), fmt.n(c.f1), fmt.n(c.accuracy)]; });
            modelOut.append(h("div", { class: "grid grid--2" },
                h("div", { class: "card" }, h("h3", {}, "Relevance classifier ", h("small", {}, "model: " + rep.chosen + " (chosen by validation PR-AUC)")), PL.charts.table(["Split", "Positive rate", "ROC-AUC", "PR-AUC", "Precision", "Recall", "F1", "Accuracy"], rows),
                    h("p", { class: "note" }, "Only about 1 in 9 candidates is relevant, so accuracy alone is misleading; PR-AUC is the informative number. This is the one place ROC-AUC applies: a real binary task with labels. Used to explain, not to rerank (no significant ranking gain on validation).")),
                h("div", { class: "card", id: "gshap" })));
            PL.charts.hbars(document.getElementById("gshap"), { title: "Global SHAP importance (mean |log-odds|, validation)", valueName: "mean |SHAP|", format: (v) => v.toFixed(2),
                items: Object.entries(rep.global_shap_importance_validation || {}).slice(0, 8).map(([k, v]) => ({ label: k.replace(/_/g, " "), value: v, color: "var(--viz-tfidf)" })),
                note: "Rank features are correlated, so credit is shared between them. SHAP/LIME explain this surrogate over named features, not the embedding dimensions." });
        } catch (e) { PL.error(modelOut, e, loadModel); }
    }

    // ---------- answer explanation ----------
    function renderAnswerExplanation(e, answer) {
        ansOut.replaceChildren(
            e.verification && e.verification.replaced ? h("div", { class: "banner banner--info" }, "The model draft was rejected (" + e.verification.reason + "); the answer below is quoted passages.") : null,
            h("div", { class: "card" }, h("h3", {}, "Answer ", h("small", {}, e.answer_mode)), ...e.sentences.map((s) => h("div", { style: "margin:.5rem 0;padding:.5rem .7rem;border-left:3px solid " + ({ stated_in_source: "var(--success)", interpretation: "#818cf8", unverified: "var(--danger)", meta: "var(--glass-border)" }[s.kind]) },
                h("div", { class: "small" }, s.text), h("div", { class: "row small" }, h("span", { class: "chip " + ({ stated_in_source: "chip--ok", unverified: "chip--bad" }[s.kind] || "") }, s.kind.replace(/_/g, " ")),
                    s.evidence_strength !== "none" || s.kind === "stated_in_source" ? h("span", { class: "strength pill-" + s.evidence_strength }, "evidence: " + s.evidence_strength) : null, s.warning ? h("span", { class: "chip chip--warn" }, s.warning) : null),
                ...s.evidence.map((ev) => h("div", { class: "ev", style: "margin-top:.4rem" }, h("div", { class: "ev__title" }, `${ev.label} · ${ev.title}`), h("div", { class: "ev__meta" }, `p. ${ev.pages[0]}${ev.pages[1] !== ev.pages[0] ? "–" + ev.pages[1] : ""} · ${ev.section.replace(/_/g, " ")}${ev.supporting ? " · best supporting passage" : ""}`),
                    h("div", { class: "ev__text is-open" }, ...highlight(ev.text, ev.highlights))))))),
            Object.keys(e.papers).length ? h("div", { class: "card" }, h("h3", {}, "Contribution by paper"), PL.charts.table(["Paper", "Cited claims", "Supported", "Share"], Object.values(e.papers).map((p) => [PL.fmt.short(p.title, 70), p.claims_cited, p.claims_supported, fmt.pct(p.share_of_cited_claims)]))) : null,
            h("p", { class: "note" }, e.note));
    }
    function highlight(text, spans) {
        const m = []; (spans || []).slice().sort((a, b) => a.start - b.start).forEach((s) => { const l = m[m.length - 1]; if (l && s.start <= l.end) l.end = Math.max(l.end, s.end); else m.push({ ...s }); });
        const out = []; let i = 0; m.forEach((s) => { out.push(document.createTextNode(text.slice(i, s.start)), h("mark", { class: "hl" }, text.slice(s.start, s.end))); i = s.end; }); out.push(document.createTextNode(text.slice(i))); return out;
    }
    async function explainAnswer(q, btn) {
        PL.loading(ansOut, "Running the RAG pipeline…"); btn.disabled = true;
        try { const d = await PL.api("/api/explain/answer", { json: { question: q } }); renderAnswerExplanation(d.explanation, d.answer); }
        catch (e) { PL.error(ansOut, e); } finally { btn.disabled = false; }
    }

    PL.views.explain = {
        init(el) {
            el.append(h("h2", { class: "view__title" }, "Explainability"), h("p", { class: "view__lead" }, "Why a passage was retrieved, and which source passages support each part of an answer. Exact facts are shown separately from model-based attributions."));
            const q = h("input", { class: "input", style: "flex:1;min-width:240px", placeholder: "e.g. How does the Swin Transformer reduce self-attention cost?", "aria-label": "Query to explain" });
            const k = h("select", { class: "input", "aria-label": "Number of results" }, [3, 5].map((n) => h("option", { value: n }, n + " results")));
            const lime = h("input", { type: "checkbox", id: "lime-on" });
            const btn = h("button", { class: "btn btn--primary", onclick: () => q.value.trim() && explainRetrieval(q.value.trim(), k.value, lime.checked, btn) }, "Explain retrieval");
            q.addEventListener("keydown", (e) => { if (e.key === "Enter") btn.click(); });
            retOut = h("div", {});
            el.append(h("h3", {}, "1 · Why was this retrieved?"), h("div", { class: "row" }, q, k, h("label", { class: "row small" }, lime, "also LIME (slower)"), btn), h("div", { class: "spacer" }), retOut);
            el.append(h("div", { class: "spacer" }), h("h3", {}, "2 · The relevance model behind SHAP and LIME"), (modelOut = h("div", {})));
            const aq = h("input", { class: "input", style: "flex:1;min-width:240px", placeholder: "Ask a question to see how its answer is grounded", "aria-label": "Question to explain" });
            const abtn = h("button", { class: "btn btn--primary", onclick: () => aq.value.trim() && explainAnswer(aq.value.trim(), abtn) }, "Explain answer");
            const last = h("button", { class: "btn btn--ghost", onclick: async () => { if (!PL.state.lastResponse) return PL.toast("Ask something in the Assistant first.", "info"); PL.loading(ansOut); try { renderAnswerExplanation(await PL.api("/api/explain/response", { json: { response: PL.state.lastResponse } })); } catch (e) { PL.error(ansOut, e); } } }, "Explain the Assistant's last answer");
            ansOut = h("div", {});
            el.append(h("div", { class: "spacer" }), h("h3", {}, "3 · How is an answer grounded?"), h("div", { class: "row" }, aq, abtn, last), h("div", { class: "spacer" }), ansOut);
            loadModel();
        },
    };
})();
