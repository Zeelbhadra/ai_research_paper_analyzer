// Browser smoke test driving the reader-facing UI through Chrome DevTools Protocol (Node >= 22, no dependencies).
//   1. start the app (public mode):   PORT=5055 python app.py
//   2. node tests/ui_smoke.mjs [baseUrl] [outDir]            (CHROME env var overrides the browser path)
// Exits non-zero if any check fails; writes screenshots to outDir. Uses the live arXiv API for the Discover checks.
import { spawn } from "node:child_process";
import { mkdtempSync, mkdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const BASE = process.argv[2] || "http://127.0.0.1:5055";
const OUT = process.argv[3] || join(tmpdir(), "ui_smoke");
const CHROME = process.env.CHROME || "C:/Program Files/Google/Chrome/Application/chrome.exe";
const PORT = 9333;
mkdirSync(OUT, { recursive: true });

const chrome = spawn(CHROME, ["--headless=new", "--disable-gpu", `--remote-debugging-port=${PORT}`, `--user-data-dir=${mkdtempSync(join(tmpdir(), "chr-"))}`, "--window-size=1280,1000", "about:blank"], { stdio: "ignore" });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let ws, id = 0; const pending = new Map(); const jsErrors = [];

async function connect() {
    for (let i = 0; i < 50; i++) {
        try {
            const pages = await (await fetch(`http://127.0.0.1:${PORT}/json`)).json();
            const p = pages.find((x) => x.type === "page");
            if (p) { ws = new WebSocket(p.webSocketDebuggerUrl); await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; }); break; }
        } catch (_) { /* not up yet */ }
        await sleep(300);
    }
    ws.onmessage = (m) => {
        const d = JSON.parse(m.data);
        if (d.id && pending.has(d.id)) { pending.get(d.id)(d); pending.delete(d.id); }
        if (d.method === "Runtime.exceptionThrown") jsErrors.push(d.params.exceptionDetails.exception?.description || d.params.exceptionDetails.text);
    };
}
const send = (method, params = {}) => new Promise((res) => { const i = ++id; pending.set(i, res); ws.send(JSON.stringify({ id: i, method, params })); });
const ev = async (expr) => { const r = await send("Runtime.evaluate", { expression: expr, returnByValue: true, awaitPromise: true }); if (r.result.exceptionDetails) throw new Error(r.result.exceptionDetails.exception?.description || "eval failed"); return r.result.result.value; };
async function waitFor(expr, ms = 60000, label = expr) { const t = Date.now(); while (Date.now() - t < ms) { try { if (await ev(expr)) return true; } catch (_) { } await sleep(400); } throw new Error("timeout waiting for: " + label); }
async function shot(name) {
    const m = await send("Page.getLayoutMetrics"); const h = Math.min(Math.ceil(m.result.cssContentSize.height), 4000);
    const r = await send("Page.captureScreenshot", { format: "png", clip: { x: 0, y: 0, width: 1280, height: h, scale: 1 }, captureBeyondViewport: true });
    writeFileSync(join(OUT, name + ".png"), Buffer.from(r.result.data, "base64"));
}
const go = async (hash) => { await send("Page.navigate", { url: `${BASE}/#/${hash}` }); await sleep(600); await send("Page.reload"); await sleep(1200); };

const results = []; const check = (name, ok, detail = "") => { results.push({ name, ok: !!ok, detail }); console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? "  -  " + detail : ""}`); };
async function step(name, fn) { try { await fn(); } catch (e) { check(name, false, String(e.message || e).slice(0, 220)); } }

// Words that would reveal how the system works. Checked on interface text (buttons, headings, notes, labels), not on paper content.
const CHROME_TEXT = `[...document.querySelectorAll('button:not(.chart__toggle), .view__title, .view__lead, .hero__title, .hero__subtitle, .note, .toast, summary, label, .section__title, h3, .header, .footer')].map(e => e.innerText).join(' | ')`;
const FORBIDDEN = `/tf-?idf|hybrid|sbert|\\be5\\b|\\bbge\\b|faiss|embedding|cosine|fallback|retriev|ndcg|latency|\\brank\\b|\\bscore\\b|provider|openai|extractive|language model|shap|lime\\b|benchmark|pipeline|dashboard/i`;
const leaks = async () => ev(`(${FORBIDDEN}).exec(${CHROME_TEXT})?.[0] || ''`);
const ANSWER_DONE = `(() => { const m = [...document.querySelectorAll('#view-assistant .msg--bot')].pop(); return !!m && !m.textContent.includes('Reading the paper') && document.querySelector('#view-assistant .composer button').disabled === false; })()`;
const ask = async (q) => { await ev(`(() => { const t = document.querySelector('#view-assistant textarea'); t.value = ${JSON.stringify(q)}; document.querySelector('#view-assistant .composer button').click(); })()`); await sleep(500); await waitFor(ANSWER_DONE, 240000, "assistant answer"); };

await connect();
await send("Page.enable"); await send("Runtime.enable");
await send("Emulation.setDeviceMetricsOverride", { width: 1280, height: 1000, deviceScaleFactor: 1, mobile: false });

await step("navigation", async () => {
    await go("discover");
    const nav = await ev(`[...document.querySelectorAll('.header__link')].map(a => a.textContent.trim())`);
    check("navigation has only Discover, Library, Assistant", JSON.stringify(nav) === JSON.stringify(["Discover", "Library", "Assistant"]), JSON.stringify(nav));
    for (const gone of ["dashboard", "benchmarks", "explain", "pipeline"]) {
        await go(gone);
        check(`#/${gone} is not a page (falls back to Discover)`, await ev(`!document.getElementById('view-discover').hidden && !document.getElementById('view-${gone}')`));
    }
    const codes = await ev(`Promise.all(['/api/dashboard', '/api/benchmarks/overview', '/api/pipeline/status', '/api/retrieval/status', '/api/explain/model'].map(u => fetch(u).then(r => r.status)))`);
    check("backend/admin endpoints are not available to readers", codes.every((c) => c === 404), JSON.stringify(codes));
});

await step("discover: search", async () => {
    await go("discover");
    await ev(`(() => { document.querySelector('#view-discover .search-input').value = 'attention in vision transformers'; document.querySelector('#view-discover .search-bar .btn--primary').click(); })()`);
    await waitFor(`document.querySelectorAll('#view-discover .paper').length > 0 || document.querySelector('#view-discover .empty')`, 120000, "search results");
    const n = await ev(`document.querySelectorAll('#view-discover .paper').length`);
    check("topic search returns papers with Analyze buttons", n >= 3 && await ev(`[...document.querySelectorAll('#view-discover .paper')].every(c => [...c.querySelectorAll('button')].some(b => b.textContent === 'Analyze'))`), `${n} papers`);
    check("no backend terms on Discover", (await leaks()) === "", await leaks());
    await shot("discover");
});

await step("discover: analyze", async () => {
    await ev(`[...document.querySelectorAll('#view-discover .paper button')].find(b => b.textContent === 'Analyze').click()`);
    await waitFor(`document.getElementById('drawer').innerText.includes('RELATED TOPICS') || document.getElementById('drawer').innerText.includes('Could not')`, 300000, "analysis");
    await waitFor(`!document.getElementById('drawer').innerText.includes('Looking')`, 120000, "similar/related lists");
    const t = await ev(`document.getElementById('drawer').innerText`);
    check("analysis shows summary, keywords, related topics, similar and more papers", ["SUMMARY", "KEYWORDS", "RELATED TOPICS", "SIMILAR PAPERS", "MORE PAPERS ON THIS TOPIC"].every((s) => t.includes(s)));
    check("analysis has no stray null/undefined", !/\b(null|undefined|NaN)\b/.test(t));
    await shot("analysis");
    await ev(`document.querySelector('#drawer .drawer__close').click()`);
});

await step("library", async () => {
    await go("library");
    await waitFor(`document.querySelectorAll('#view-library .paper').length > 0`, 20000, "library cards");
    check("library lists papers without technical labels", (await leaks()) === "" && !/chunks|processed|failed/i.test(await ev(`document.getElementById('view-library').innerText`)));
    await ev(`document.querySelector('#view-library .paper').click()`);
    await waitFor(`document.getElementById('drawer').innerText.includes('Understand this paper')`, 20000, "paper view");
    const btns = await ev(`[...document.querySelectorAll('#drawer .chip--accent')].map(b => b.textContent)`);
    check("paper view offers one-click plain-language explanations", btns.includes("Explain it simply") && btns.length >= 5, JSON.stringify(btns));
    await shot("library_paper");
    await ev(`[...document.querySelectorAll('#drawer .chip--accent')].find(b => b.textContent === 'Explain it simply').click()`);
    await waitFor(`location.hash === '#/assistant'`, 10000, "navigate to assistant");
    await sleep(800);
    await waitFor(ANSWER_DONE, 300000, "explanation");
    const info = await ev(`({ scope: document.querySelector('#view-assistant .chip--accent')?.textContent || '', bubbles: document.querySelectorAll('#view-assistant .msg--bot').length, srcs: document.querySelectorAll('#view-assistant .ev').length, text: [...document.querySelectorAll('#view-assistant .msg--bot')].pop().innerText })`);
    check("'Explain it simply' opens the assistant on that paper and answers", info.scope.length > 3 && info.text.length > 60, info.scope);
    check("answer lists its sources with pages", info.srcs >= 1 && /page/.test(await ev(`document.getElementById('view-assistant').innerHTML`)), `${info.srcs} sources`);
    await shot("assistant_explain");
});

await step("assistant", async () => {
    await ask("What do the technical terms mean?");
    check("assistant answers a follow-up about the same paper", await ev(`document.querySelectorAll('#view-assistant .msg--bot').length >= 2`));
    check("suggested follow-up questions are offered", await ev(`document.querySelectorAll('#view-assistant .suggest .chip').length >= 2`));
    const cite = await ev(`document.querySelectorAll('#view-assistant .msg--bot .cite').length`);
    if (cite > 0) { await ev(`document.querySelector('#view-assistant .msg--bot .cite').click()`); check("clicking a citation reveals its source", await ev(`!!document.querySelector('#view-assistant .ev.is-active')`)); }
    await ev(`document.querySelector('#view-assistant .scope-clear, #view-assistant .chip--accent')?.click()`);          // widen to the whole library
    await ask("What is the best way to bake sourdough bread?");
    check("off-topic question gets a polite 'not found in the papers'", await ev(`[...document.querySelectorAll('#view-assistant .msg--bot')].pop().innerText.toLowerCase().includes('could not find enough information')`));
    const txt = await ev(`document.getElementById('view-assistant').innerText`);
    check("no stray null/undefined in the conversation", !/\b(null|undefined|NaN)\b/.test(txt));
    check("no backend terms in the assistant interface", (await leaks()) === "", await leaks());
    check("answers never mention how they were produced", !/extractive|language model|fallback|retriev|provider/i.test(txt));
    await shot("assistant");
});

check("no JavaScript errors", jsErrors.length === 0, jsErrors.slice(0, 2).join(" | "));
const failed = results.filter((r) => !r.ok);
writeFileSync(join(OUT, "results.json"), JSON.stringify({ results, jsErrors }, null, 1));
console.log(`\n${results.length - failed.length}/${results.length} checks passed; screenshots in ${OUT}`);
ws.close(); chrome.kill();
process.exit(failed.length ? 1 : 0);
