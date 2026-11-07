// Browser smoke test driving the real UI through Chrome DevTools Protocol (Node >= 22, no dependencies).
//   1. start the app:   PORT=5055 python app.py
//   2. node tests/ui_smoke.mjs [baseUrl] [outDir]            (CHROME env var overrides the browser path)
// Exits non-zero if any check fails; writes screenshots to outDir.
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

// An answer is complete when the newest assistant bubble has its metadata line (or an error alert) and the composer is free.
const ANSWER_DONE = `(() => { const m = [...document.querySelectorAll('#view-assistant .msg--bot')].pop(); return !!m && (!!m.querySelector('.msg__meta') || !!m.querySelector('[role=alert]')) && document.querySelector('#view-assistant .composer button').disabled === false; })()`;
const results = []; const check = (name, ok, detail = "") => { results.push({ name, ok: !!ok, detail }); console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? "  -  " + detail : ""}`); };
async function step(name, fn) { try { await fn(); } catch (e) { check(name, false, String(e.message || e).slice(0, 200)); } }

await connect();
await send("Page.enable"); await send("Runtime.enable");
await send("Emulation.setDeviceMetricsOverride", { width: 1280, height: 1000, deviceScaleFactor: 1, mobile: false });

await step("every route renders without JS errors", async () => {
    const stray = [];
    for (const r of ["discover", "dashboard", "library", "benchmarks", "pipeline", "explain", "assistant"]) {
        await go(r); await sleep(2500);
        const bad = await ev(`(document.getElementById('view-${r}').innerText.match(/\b(null|undefined|NaN)\b/g) || []).length`);
        if (bad) stray.push(r + ":" + bad);
    }
    check("no stray 'null'/'undefined'/'NaN' text on any view", stray.length === 0, stray.join(", "));
    check("every route renders without JS errors", jsErrors.length === 0, jsErrors.slice(0, 2).join(" | "));
});

await step("assistant: grounded answer with citations and evidence", async () => {
    await go("assistant");
    await ev(`(() => { const t = document.querySelector('#view-assistant textarea'); t.value = 'How does the Swin Transformer reduce the computational complexity of self-attention?'; document.querySelector('#view-assistant .composer button').click(); })()`);
    await waitFor(ANSWER_DONE, 240000, "assistant final answer");
    const info = await ev(`(() => ({ chips: [...document.querySelectorAll('#view-assistant .msg--bot:last-child .chip')].map(c => c.textContent).slice(0,4), cites: document.querySelectorAll('#view-assistant .msg--bot .cite').length, ev: document.querySelectorAll('#view-assistant .ev').length, marks: document.querySelectorAll('#view-assistant mark.hl').length, meta: document.querySelector('#view-assistant .msg__meta')?.textContent.slice(0,200) }))()`);
    check("assistant returns an answer", info.chips.length > 0, JSON.stringify(info.chips));
    check("assistant answer has no stray 'null' text", await ev(`!/\b(null|undefined|NaN)\b/.test(document.getElementById('view-assistant').innerText)`));
    check("assistant shows evidence cards with pages", info.ev >= 3, `${info.ev} evidence cards, ${info.marks} highlights, ${info.cites} citation chips`);
    if (info.cites > 0) {
        await ev(`document.querySelector('#view-assistant .msg--bot .cite').click()`);
        check("clicking a citation highlights its evidence card", await ev(`!!document.querySelector('#view-assistant .ev.is-active')`));
    }
    await shot("assistant_answer");
});

await step("assistant: off-topic question is refused", async () => {
    await ev(`(() => { const t = document.querySelector('#view-assistant textarea'); t.value = 'What is the best way to bake sourdough bread?'; document.querySelector('#view-assistant .composer button').click(); })()`);
    await waitFor(ANSWER_DONE, 90000, "off-topic answer");
    check("off-topic question shows 'Insufficient evidence'", await ev(`[...document.querySelectorAll('#view-assistant .msg--bot')].pop().textContent.includes('Insufficient evidence')`));
});

await step("assistant: follow-up is recognised", async () => {
    await ev(`(() => { const t = document.querySelector('#view-assistant textarea'); t.value = 'How does the Swin Transformer reduce self-attention cost?'; document.querySelector('#view-assistant .composer button').click(); })()`);
    await waitFor(ANSWER_DONE, 240000, "answer 3");
    await ev(`(() => { const t = document.querySelector('#view-assistant textarea'); t.value = 'What about its accuracy?'; document.querySelector('#view-assistant .composer button').click(); })()`);
    await waitFor(ANSWER_DONE, 240000, "follow-up answer");
    check("follow-up question is flagged as a follow-up", await ev(`[...document.querySelectorAll('#view-assistant .msg--bot')].pop().textContent.includes('follow-up')`));
});

await step("explainability: SHAP and LIME for a query", async () => {
    await go("explain");
    await ev(`(() => { document.querySelector('#view-explain input.input').value = 'How does the Swin Transformer reduce self-attention cost?'; document.getElementById('lime-on').checked = true; document.querySelector('#view-explain .btn--primary').click(); })()`);
    await waitFor(`document.querySelectorAll('#view-explain .result-ex').length >= 1`, 180000, "explain results");
    const n = await ev(`({ cards: document.querySelectorAll('#view-explain .result-ex').length, bars: document.querySelectorAll('#view-explain .result-ex .dbar').length, gauges: document.querySelectorAll('#view-explain .gauge').length })`);
    check("retrieval explanation shows SHAP/LIME bars", n.cards >= 1 && n.bars >= 6, JSON.stringify(n));
    await shot("explain");
});

await step("library: open a paper", async () => {
    await go("library");
    await waitFor(`document.querySelectorAll('#view-library .paper').length > 0`, 20000, "library cards");
    await ev(`document.querySelector('#view-library .paper').click()`);
    await waitFor(`!document.getElementById('drawer').hidden && document.querySelectorAll('#drawer details.sec').length > 0`, 20000, "drawer");
    check("paper drawer shows sections with page numbers", await ev(`document.getElementById('drawer').textContent.includes('pp.') || document.getElementById('drawer').textContent.includes('p. ')`));
    await shot("library_drawer");
});

await step("pipeline: trigger a report", async () => {
    await go("pipeline");
    await waitFor(`document.querySelector('#view-pipeline select')`, 20000, "pipeline form");
    await ev(`(() => { const s = document.querySelector('#view-pipeline select'); s.value = 'report'; document.querySelector('#view-pipeline .btn--primary').click(); })()`);
    await waitFor(`document.querySelector('.toast')?.textContent.includes('Started')`, 15000, "toast");
    await sleep(3000); await ev(`location.reload()`); await sleep(2500);
    check("report workflow ran and appears in the job list", await ev(`document.getElementById('view-pipeline').textContent.includes('report')`));
    await shot("pipeline");
});

await step("legacy: search and analyze", async () => {
    await go("discover");
    await ev(`(() => { document.getElementById('search-input').value = 'attention is all you need'; document.querySelector('input[name=source][value=arxiv]').checked = true; document.getElementById('search-form').requestSubmit(); })()`);
    await waitFor(`document.querySelectorAll('#search-results .paper-card').length > 0`, 90000, "search results");
    check("legacy search returns cards with abstracts", await ev(`document.querySelector('#search-results .paper-card .paper-card__abstract').textContent.length > 20`));
    await ev(`document.querySelector('.analyze-paper-btn').click()`);
    await waitFor(`!document.getElementById('analysis-section').hidden`, 240000, "analysis");
    const a = await ev(`({ sum: document.getElementById('summary-text').textContent.length, kw: document.querySelectorAll('#keywords-container .tag').length, topics: document.querySelectorAll('#topics-container .topic-bar').length })`);
    check("legacy analyze returns summary, keywords and topics", a.sum > 40 && a.kw > 0 && a.topics > 0, JSON.stringify(a));
    await shot("discover");
});

const failed = results.filter((r) => !r.ok);
writeFileSync(join(OUT, "results.json"), JSON.stringify({ results, jsErrors }, null, 1));
console.log(`\n${results.length - failed.length}/${results.length} checks passed; JS errors: ${jsErrors.length}; screenshots in ${OUT}`);
ws.close(); chrome.kill();
process.exit(failed.length ? 1 : 0);
