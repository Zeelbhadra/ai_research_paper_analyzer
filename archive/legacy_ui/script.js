

(function () {
    "use strict";

    // ---- DOM refs ---------------------------------------------------
    const searchForm = document.getElementById("search-form");
    const searchInput = document.getElementById("search-input");
    const searchBtn = document.getElementById("search-btn");
    const resultsSection = document.getElementById("results-section");
    const resultCount = document.getElementById("result-count");
    const searchResults = document.getElementById("search-results");

    const uploadZone = document.getElementById("upload-zone");
    const fileInput = document.getElementById("file-input");
    const textInput = document.getElementById("text-input");
    const analyzeBtn = document.getElementById("analyze-btn");

    const analysisSection = document.getElementById("analysis-section");
    const summaryText = document.getElementById("summary-text");
    const keywordsContainer = document.getElementById("keywords-container");
    const topicsContainer = document.getElementById("topics-container");
    const similarContainer = document.getElementById("similar-container");

    const toastContainer = document.getElementById("toast-container");

    // ---- Utilities --------------------------------------------------
    function showToast(message, type = "info") {
        const el = document.createElement("div");
        el.className = `toast toast--${type}`;
        el.textContent = message;
        toastContainer.appendChild(el);
        setTimeout(() => el.remove(), 4000);
    }

    function setLoading(btn, loading) {
        const text = btn.querySelector(".btn__text");
        const spinner = btn.querySelector(".btn__spinner");
        if (loading) {
            text.hidden = true;
            spinner.hidden = false;
            btn.disabled = true;
        } else {
            text.hidden = false;
            spinner.hidden = true;
            btn.disabled = false;
        }
    }

    function truncate(str, len = 200) {
        if (!str) return "";
        return str.length > len ? str.slice(0, len) + "…" : str;
    }

    // ---- Search -----------------------------------------------------
    searchForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const query = searchInput.value.trim();
        if (!query) return;

        const source = document.querySelector('input[name="source"]:checked').value;
        setLoading(searchBtn, true);
        searchResults.innerHTML = "";
        resultsSection.hidden = true;

        try {
            const resp = await fetch(`/api/search?q=${encodeURIComponent(query)}&source=${source}&limit=12`);
            const data = await resp.json();

            if (!resp.ok) {
                showToast(data.error || "Search failed", "error");
                return;
            }

            renderSearchResults(data.results);
            resultCount.textContent = data.count;
            resultsSection.hidden = false;
            resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
        } catch (err) {
            showToast("Network error — is the server running?", "error");
        } finally {
            setLoading(searchBtn, false);
        }
    });

    function renderSearchResults(papers) {
        searchResults.innerHTML = "";
        papers.forEach((p, i) => {
            const card = document.createElement("div");
            card.className = "paper-card";
            card.style.animationDelay = `${i * 0.06}s`;

            const sourceClass = p.source === "arxiv" ? "paper-card__source--arxiv" : "paper-card__source--semantic_scholar";
            const sourceLabel = p.source === "arxiv" ? "arXiv" : "Semantic Scholar";

            card.innerHTML = `
                <span class="paper-card__source ${sourceClass}">${sourceLabel}</span>
                <div class="paper-card__title">${escapeHtml(p.title)}</div>
                <div class="paper-card__authors">${escapeHtml((p.authors || []).slice(0, 4).join(", "))}${(p.authors || []).length > 4 ? " et al." : ""}</div>
                <div class="paper-card__abstract">${escapeHtml(truncate(p.abstract, 260))}</div>
                <div class="paper-card__actions">
                    <button class="btn btn--small analyze-paper-btn" data-abstract="${escapeAttr(p.abstract || "")}">Analyze</button>
                    ${p.url ? `<a href="${escapeAttr(p.url)}" target="_blank" class="btn btn--small" style="background:rgba(255,255,255,0.08);color:var(--text-secondary);">View</a>` : ""}
                </div>
            `;
            searchResults.appendChild(card);
        });

        // Attach analyze handlers
        document.querySelectorAll(".analyze-paper-btn").forEach((btn) => {
            btn.addEventListener("click", () => {
                const abstract = btn.dataset.abstract;
                if (abstract) {
                    textInput.value = abstract;
                    analyzeText(abstract);
                }
            });
        });
    }

    // ---- Upload / Analyze -------------------------------------------
    // Drag-and-drop
    uploadZone.addEventListener("click", () => fileInput.click());
    uploadZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        uploadZone.classList.add("dragover");
    });
    uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("dragover"));
    uploadZone.addEventListener("drop", (e) => {
        e.preventDefault();
        uploadZone.classList.remove("dragover");
        const files = e.dataTransfer.files;
        if (files.length && files[0].type === "application/pdf") {
            fileInput.files = files;
            analyzePDF(files[0]);
        } else {
            showToast("Please drop a PDF file.", "error");
        }
    });
    fileInput.addEventListener("change", () => {
        if (fileInput.files.length) {
            const name = fileInput.files[0].name;
            uploadZone.querySelector(".upload-zone__text").innerHTML =
                `<strong>${escapeHtml(name)}</strong><br>Click Analyze to process`;
        }
    });

    analyzeBtn.addEventListener("click", () => {
        if (fileInput.files && fileInput.files.length) {
            analyzePDF(fileInput.files[0]);
        } else if (textInput.value.trim()) {
            analyzeText(textInput.value.trim());
        } else {
            showToast("Please upload a PDF or paste some text first.", "error");
        }
    });

    async function analyzePDF(file) {
        setLoading(analyzeBtn, true);
        showToast("Uploading & analyzing PDF — this may take a moment while models load…", "info");
        const form = new FormData();
        form.append("file", file);

        try {
            const resp = await fetch("/api/analyze", { method: "POST", body: form });
            const data = await resp.json();
            if (!resp.ok) {
                showToast(data.error || "Analysis failed", "error");
                return;
            }
            renderAnalysis(data);
            // Also search for similar papers using the summary
            if (data.summary) fetchSimilar(data.summary);
        } catch (err) {
            showToast("Network error during analysis.", "error");
        } finally {
            setLoading(analyzeBtn, false);
        }
    }

    async function analyzeText(text) {
        setLoading(analyzeBtn, true);
        try {
            const resp = await fetch("/api/analyze", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text }),
            });
            const data = await resp.json();
            if (!resp.ok) {
                showToast(data.error || "Analysis failed", "error");
                return;
            }
            renderAnalysis(data);

            // Also fetch similar papers
            fetchSimilar(text);
        } catch (err) {
            showToast("Network error during analysis.", "error");
        } finally {
            setLoading(analyzeBtn, false);
        }
    }

    async function fetchSimilar(text) {
        try {
            const resp = await fetch("/api/similar", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text }),
            });
            const data = await resp.json();
            if (resp.ok && data.similar) {
                renderSimilar(data.similar);
            }
        } catch (_) { /* silent */ }
    }

    // ---- Render Analysis --------------------------------------------
    function renderAnalysis(data) {
        // Summary
        summaryText.textContent = data.summary || "No summary generated.";

        // Keywords
        keywordsContainer.innerHTML = "";
        (data.keywords || []).forEach((kw) => {
            const tag = document.createElement("span");
            tag.className = "tag";
            tag.innerHTML = `${escapeHtml(kw.word)} <span class="tag__score">${(kw.score * 100).toFixed(0)}%</span>`;
            keywordsContainer.appendChild(tag);
        });

        // Topics
        topicsContainer.innerHTML = "";
        (data.topics || []).forEach((t) => {
            const pct = (t.score * 100).toFixed(1);
            const bar = document.createElement("div");
            bar.className = "topic-bar";
            bar.innerHTML = `
                <span class="topic-bar__label">${escapeHtml(t.label)}</span>
                <div class="topic-bar__track">
                    <div class="topic-bar__fill" style="width: 0%"></div>
                </div>
                <span class="topic-bar__pct">${pct}%</span>
            `;
            topicsContainer.appendChild(bar);
            // Animate bar fill
            requestAnimationFrame(() => {
                bar.querySelector(".topic-bar__fill").style.width = `${pct}%`;
            });
        });

        // Show section
        analysisSection.hidden = false;
        analysisSection.scrollIntoView({ behavior: "smooth", block: "start" });
        showToast("Analysis complete!", "success");
    }

    function renderSimilar(items) {
        similarContainer.innerHTML = "";
        if (!items.length) {
            similarContainer.innerHTML = '<p style="color:var(--text-muted);font-size:0.9rem;">No similar papers in index yet. Search for more papers first!</p>';
            return;
        }
        items.forEach((item) => {
            const el = document.createElement("div");
            el.className = "similar-item";
            const title = item.paper?.title || "Untitled";
            const url = item.paper?.url || "#";
            const sim = item.similarity != null ? item.similarity.toFixed(3) : "—";
            el.innerHTML = `
                <a href="${escapeAttr(url)}" target="_blank" class="similar-item__title">${escapeHtml(truncate(title, 80))}</a>
                <span class="similar-item__badge" title="cosine similarity under the production embedding model">sim ${sim}</span>
            `;
            similarContainer.appendChild(el);
        });
    }

    // ---- Helpers -----------------------------------------------------
    function escapeHtml(str) {
        const div = document.createElement("div");
        div.appendChild(document.createTextNode(str));
        return div.innerHTML;
    }
    function escapeAttr(str) {
        return str.replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/'/g, "&#39;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }
})();
