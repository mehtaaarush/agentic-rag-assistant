// Research agent. The server runs the Gemini loop; this page embeds the agent's paper
// queries with all-MiniLM-L6-v2 and ranks the 1,272 chunks itself.
const TRANSFORMERS = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";
const TOP_K = 4;

const $ = (s) => document.querySelector(s);
const nf = new Intl.NumberFormat("en-GB");
const SHORT = {
  attention_is_all_you_need: "Transformer", vision_transformer_vit: "ViT", swin_transformer: "Swin Transformer", swinir: "SwinIR",
  dinov2: "DINOv2", dino_self_supervised_vit: "DINO", masked_autoencoders_mae: "MAE", generative_adversarial_networks: "GANs",
  clip: "CLIP", resnet: "ResNet", loftr: "LoFTR", ddpm_diffusion: "DDPM",
};

const S = { index: null, q8: null, scale: null, papers: [], model: null, modelLoading: null, contents: [], busy: false, turns: [] };

/* ---------- data ---------- */
async function loadIndex() {
  const [index, buf, papers, evals] = await Promise.all([
    fetch("data/index.json").then((r) => r.json()),
    fetch("data/embeddings.bin").then((r) => r.arrayBuffer()),
    fetch("data/papers.json").then((r) => r.json()),
    fetch("data/eval.json").then((r) => r.json()),
  ]);
  S.index = index;
  S.q8 = new Int8Array(buf, 0, index.count * index.dim);
  S.scale = new Float32Array(buf, index.count * index.dim, index.count);
  S.papers = papers.map((p) => ({ ...p, short: SHORT[p.file.replace(/\.pdf$/, "")] || p.title }));
  renderPapers();
  renderEval(evals);
}

function renderPapers() {
  $("#t-papers").innerHTML = S.papers.map((p) => `<tr><td>${esc(p.title)}</td><td class="num">${p.year}</td><td class="num">${p.pages}</td><td class="num">${p.chunks}</td><td><a href="https://arxiv.org/abs/${p.arxiv}">arXiv ${p.arxiv}</a></td></tr>`).join("");
  const total = S.papers.reduce((a, p) => a + p.chunks, 0);
  $("#d-papers").textContent = `${nf.format(total)} chunks of 1,000 characters with a 200-character overlap, embedded with all-MiniLM-L6-v2.`;
  $("#m-chunks").textContent = nf.format(total);
}

function renderEval(e) {
  const f = (v) => (v == null ? "not scored" : v.toFixed(2));
  $("#e-faith").textContent = e.average_faithfulness.toFixed(2);
  $("#e-rel").textContent = e.average_relevancy.toFixed(2);
  $("#t-eval").innerHTML = e.per_question.map((q) => `<tr><td>${esc(q.question)}</td><td class="num">${f(q.faithfulness)}</td><td class="num">${f(q.relevancy)}</td></tr>`).join("");
}

/* ---------- the embedding model ---------- */
function loadModel() {
  if (S.model) return Promise.resolve(S.model);
  if (S.modelLoading) return S.modelLoading;
  const files = new Map();
  S.modelLoading = (async () => {
    const { pipeline, env } = await import(TRANSFORMERS);
    env.allowLocalModels = false;
    S.model = await pipeline("feature-extraction", S.index.model, {
      dtype: S.index.dtype,
      progress_callback: (p) => {
        if (p.status !== "progress" || !p.total) return;
        files.set(p.file, [p.loaded, p.total]);
        let a = 0, b = 0;
        for (const [l, t] of files.values()) { a += l; b += t; }
        if (S.busy) status(`Loading the search model, ${Math.round(b / 1e6)} MB`, a / b);
      },
    });
    return S.model;
  })();
  S.modelLoading.catch(() => { S.modelLoading = null; });
  return S.modelLoading;
}

async function search(query, k) {
  const model = await loadModel();
  const v = (await model(query, { pooling: "mean", normalize: true })).data;
  const { count, dim } = S.index;
  const scores = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    let d = 0;
    const o = i * dim;
    for (let j = 0; j < dim; j++) d += S.q8[o + j] * v[j];
    scores[i] = d * S.scale[i];
  }
  const order = Array.from(scores.keys()).sort((a, b) => scores[b] - scores[a]).slice(0, k);
  return order.map((i) => ({ i, score: scores[i], paper: S.index.paper[i], page: S.index.page[i] + 1 }));
}

/* ---------- asking ---------- */
async function post(body) {
  const r = await fetch("/api/agent", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  let j = null;
  try { j = await r.json(); } catch { j = { error: "The server sent a reply this page could not read." }; }
  return { ok: r.ok, status: r.status, j };
}

async function ask(question) {
  if (S.busy || !S.index) return;
  question = question.trim();
  if (!question) return;
  setBusy(true);
  const before = S.contents;
  const turn = { question, steps: [], answer: null, error: null, offline: false, t0: performance.now() };
  S.turns.push(turn);
  renderThread();
  loadModel().catch(() => {});
  let payload = { contents: S.contents, question };
  try {
    for (let hop = 0; hop < 8; hop++) {
      status("Gemini is choosing a tool");
      const { ok, j } = await post(payload);
      if (!ok) {
        if (j.code === "no_key") { await offline(turn); break; }
        mergeTrace(turn, j.trace);
        turn.error = j.error || "The agent did not answer.";
        S.contents = before;
        break;
      }
      mergeTrace(turn, j.trace);
      S.contents = j.contents;
      if (j.type === "answer") { turn.answer = j.text || "The agent returned an empty answer."; break; }
      const retrieved = [];
      for (const q of j.queries) {
        const step = { tool: "papers", query: q.query, hits: [], live: true };
        turn.steps.push(step);
        renderRoute(turn);
        status("Searching the papers in this browser");
        step.hits = await search(q.query, TOP_K);
        step.live = false;
        renderRoute(turn);
        retrieved.push({ id: q.id, query: q.query, ids: step.hits.map((h) => h.i) });
      }
      payload = { contents: j.contents, partial: j.partial, retrieved };
    }
    if (!turn.answer && !turn.error && !turn.offline) turn.error = "The agent took too many steps. Try a narrower question.";
  } catch (err) {
    turn.error = navigator.onLine ? "Something went wrong reaching the agent. Try again." : "You appear to be offline.";
    S.contents = before;
    console.error(err);
  }
  turn.ms = performance.now() - turn.t0;
  setBusy(false);
  status("");
  renderThread();
  renderRoute(turn);
}

function mergeTrace(turn, trace) {
  for (const t of trace || []) {
    if (t.tool === "search_web") {
      turn.steps.push({ tool: "web", query: t.query, engine: t.engine, sources: t.sources || [], error: t.error });
    } else if (t.tool === "search_papers") {
      const step = turn.steps.find((s) => s.tool === "papers" && !s.excerpts && s.query === t.query);
      if (step) step.excerpts = new Map(t.sources.map((s) => [s.id, s.excerpt]));
    }
  }
  renderRoute(turn);
}

// No key on the server: run the paper search on the question itself and show the passages.
async function offline(turn) {
  turn.offline = true;
  status("Searching the papers in this browser");
  const step = { tool: "papers", query: turn.question, hits: [], live: true };
  turn.steps.push(step);
  renderRoute(turn);
  step.hits = await search(turn.question, 5);
  step.live = false;
  const { ok, j } = await post({ excerpts: step.hits.map((h) => h.i) });
  if (ok) step.excerpts = new Map(j.sources.map((s) => [s.id, s.excerpt]));
}

/* ---------- rendering ---------- */
function esc(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function inline(s) {
  return esc(s)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*(?!\s)([^*]+?)\*(?!\w)/g, "$1<em>$2</em>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" rel="noopener noreferrer" target="_blank">$1</a>');
}

function md(src) {
  const out = [];
  let list = null;
  const close = () => { if (list) { out.push(`<${list.tag}>${list.items.map((i) => `<li>${inline(i)}</li>`).join("")}</${list.tag}>`); list = null; } };
  for (const raw of src.replace(/\r/g, "").split("\n")) {
    const line = raw.trimEnd();
    const ul = line.match(/^\s*[*-]\s+(.*)$/), ol = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (ul || ol) {
      const tag = ul ? "ul" : "ol";
      if (!list || list.tag !== tag) { close(); list = { tag, items: [] }; }
      list.items.push((ul || ol)[1]);
      continue;
    }
    if (list && /^\s{2,}\S/.test(line)) { list.items[list.items.length - 1] += ` ${line.trim()}`; continue; }
    close();
    if (!line.trim()) continue;
    const h = line.match(/^#{1,6}\s+(.*)$/);
    out.push(h ? `<p><strong>${inline(h[1])}</strong></p>` : `<p>${inline(line)}</p>`);
  }
  close();
  return out.join("");
}

function renderThread() {
  const el = $("#thread");
  el.innerHTML = S.turns.map((t, k) => {
    let body;
    if (t.answer != null) body = `<div class="answer">${md(t.answer)}</div>`;
    else if (t.error) body = `<p class="notice">${esc(t.error)}</p>`;
    else if (t.offline) body = offlineBody(t);
    else body = `<div class="answer is-pending"><p>Working on it. The steps appear in the route log as the agent takes them.</p></div>`;
    const calls = t.steps.length;
    const meta = t.ms != null && (t.answer != null) ? `<p class="meta">Answered in ${(t.ms / 1000).toFixed(1)} s after ${calls} tool ${calls === 1 ? "call" : "calls"}.</p>` : "";
    return `<article class="turn sheet" aria-label="Question ${k + 1}"><h3 class="asked">${esc(t.question)}</h3>${body}${meta}</article>`;
  }).reverse().join("");
  $("#reset").hidden = S.turns.length === 0;
}

function offlineBody(t) {
  const step = t.steps[t.steps.length - 1];
  const items = (step?.hits || []).map((h) => {
    const p = S.papers[h.paper];
    const ex = step.excerpts?.get(h.i);
    return `<li><b>${esc(p.short)}, page ${h.page}</b> <span class="num">${h.score.toFixed(3)}</span>${ex ? `<br>${esc(ex)}` : ""}</li>`;
  }).join("");
  return `<p class="notice">The live agent is offline right now, so here is what the paper search found for your question.</p><ol class="passages">${items}</ol>`;
}

function renderRoute(turn) {
  const log = $("#route-log");
  if (!turn || !turn.steps.length) {
    log.innerHTML = `<li class="route-empty">${turn ? "The agent is deciding which tool to use." : "Ask a question to see the agent choose its tools."}</li>`;
    if (turn && turn.answer != null && !turn.steps.length) log.innerHTML = `<li class="route-empty">The agent answered without calling a tool.</li>`;
    return;
  }
  log.innerHTML = turn.steps.map((s) => {
    if (s.tool === "web") {
      const src = s.sources.length
        ? `<ul class="passages">${s.sources.map((w) => `<li><a href="${esc(w.url)}" rel="noopener noreferrer" target="_blank">${esc(w.title)}</a></li>`).join("")}</ul>`
        : `<p class="small muted">${s.error ? "Web search failed for this query." : "No sources listed."}</p>`;
      return `<li class="step"><div class="step-head"><span class="step-tool">search_web</span><span class="step-where">${esc(s.engine || "web")}</span></div><p class="step-query">${esc(s.query)}</p>${src}</li>`;
    }
    const rows = s.hits.map((h, k) => {
      const p = S.papers[h.paper];
      const ex = s.excerpts?.get(h.i);
      return `<tr class="${k === 0 ? "top" : ""}"><td>${esc(p.short)}, page ${h.page}${ex ? `<span class="ex">${esc(ex)}</span>` : ""}</td><td class="num">${h.score.toFixed(3)}</td></tr>`;
    }).join("");
    return `<li class="step${s.live ? " is-live" : ""}"><div class="step-head"><span class="step-tool">search_papers</span><span class="step-where">in this browser</span></div><p class="step-query">${esc(s.query)}</p>${rows ? `<table class="hits"><tbody>${rows}</tbody></table>` : ""}</li>`;
  }).join("");
}

function status(text, frac) {
  const el = $("#status");
  if (frac == null) { el.textContent = text; return; }
  el.innerHTML = `${esc(text)}<span class="meter" aria-hidden="true"><i style="transform: scaleX(${Math.max(0.02, Math.min(1, frac)).toFixed(3)})"></i></span>`;
}

function setBusy(b) {
  S.busy = b;
  $("#send").disabled = b;
  $("#q").disabled = b;
  document.querySelectorAll(".try").forEach((t) => { t.disabled = b; });
  $("#send").textContent = b ? "Working" : "Ask the agent";
}

/* ---------- wiring ---------- */
$("#ask-form").addEventListener("submit", (e) => { e.preventDefault(); ask($("#q").value); });
$("#q").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); ask($("#q").value); } });
$("#q").addEventListener("focus", () => { if (S.index) loadModel().catch(() => {}); }, { once: true });
$("#tries").addEventListener("click", (e) => {
  const b = e.target.closest(".try");
  if (!b) return;
  $("#q").value = b.textContent;
  ask(b.textContent);
});
$("#reset").addEventListener("click", () => { S.turns = []; S.contents = []; renderThread(); renderRoute(null); $("#q").focus(); });

const shared = new URLSearchParams(location.search).get("q");
if (shared) $("#q").value = shared.slice(0, 600);
loadIndex().catch((err) => {
  console.error(err);
  $("#t-papers").innerHTML = `<tr><td colspan="5">The paper index did not load. Reload the page to try again.</td></tr>`;
  status("The paper index did not load. Reload the page to try again.");
  $("#send").disabled = true;
});
