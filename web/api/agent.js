// The research agent as a Vercel function: one call runs the Gemini ReAct loop until it
// answers, or until it asks for search_papers. Paper retrieval runs in the visitor's browser
// (MiniLM embeddings over the chunk index), which posts the chunk ids back; the chunk text
// lives only here. search_web runs here: Tavily when TAVILY_API_KEY is set, otherwise
// Gemini's own Google Search grounding, so the Gemini key is the only one required.
import { readFileSync } from "node:fs";
import { ApiError, GoogleGenAI } from "@google/genai";

const corpus = JSON.parse(readFileSync(new URL("../lib/corpus.json", import.meta.url), "utf8"));
const MODEL = process.env.GEMINI_MODEL || "gemini-flash-latest";
const apiKey = process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY;
const ai = apiKey ? new GoogleGenAI({ apiKey }) : null;

const MAX_ROUNDS = 6;
const MAX_QUESTION = 600;
const MAX_BODY = 400_000;
const MAX_IDS = 6;
const PER_MINUTE = 24;
const RETRY = { attempts: 3, initialDelay: 1, maxDelay: 4, httpStatusCodes: [500, 502, 503, 504] };

const SYSTEM_PROMPT = `You are a research assistant specializing in computer vision and deep learning.

You have two tools:
1. search_papers searches a local library of 12 ML research papers: Attention Is All You Need, ViT, Swin Transformer, SwinIR, DINO, DINOv2, MAE, GANs, CLIP, ResNet, LoFTR and DDPM. Prefer this for questions about model architectures, methods, and concepts.
2. search_web searches the live web. Use this only when the papers likely don't cover the topic (recent news, models released after the papers, current events).

Always decide which tool fits the question best, and use both when a question needs both. After getting results, answer clearly and cite your sources: the paper title and page for papers, the site for web results. If neither tool gives a useful answer, say so honestly. Answer only from what the tools returned.`;

const TOOLS = [{
  functionDeclarations: [
    {
      name: "search_papers",
      description: "Search the local machine-learning research papers for information. Use this for questions about model architectures, methods, and concepts covered in the papers (transformers, ViT, Swin, SwinIR, DINO, DINOv2, MAE, GANs, CLIP, ResNet, LoFTR, diffusion).",
      parametersJsonSchema: { type: "object", properties: { query: { type: "string", description: "A focused search query" } }, required: ["query"] },
    },
    {
      name: "search_web",
      description: "Search the live web for current information. Use this for recent developments, news, or anything not covered in the local research papers.",
      parametersJsonSchema: { type: "object", properties: { query: { type: "string", description: "A focused web search query" } }, required: ["query"] },
    },
  ],
}];

// Best-effort per-visitor limit; each instance keeps its own count.
const hits = new Map();
function limited(ip) {
  const now = Date.now();
  const list = (hits.get(ip) ?? []).filter((t) => now - t < 60_000);
  list.push(now);
  hits.set(ip, list);
  if (hits.size > 5000) hits.clear();
  return list.length > PER_MINUTE;
}

function papersResult(ids) {
  const clean = [...new Set(ids)].filter((i) => Number.isInteger(i) && i >= 0 && i < corpus.chunks.length).slice(0, MAX_IDS);
  const chunks = clean.map((i) => ({ i, ...corpus.chunks[i] }));
  const text = chunks.length
    ? chunks.map((c) => `[Source: ${corpus.papers[c.paper].title}, page ${c.page + 1}]\n${c.text}`).join("\n\n---\n\n")
    : "No relevant information found in the papers.";
  const sources = chunks.map((c) => ({ id: c.i, paper: c.paper, page: c.page + 1, excerpt: excerpt(c.text) }));
  return { text, sources };
}

function excerpt(text) {
  const t = text.replace(/\s+/g, " ").trim();
  return t.length > 260 ? `${t.slice(0, 257).replace(/\s\S*$/, "")} ...` : t;
}

async function searchWeb(query) {
  if (process.env.TAVILY_API_KEY) {
    const r = await fetch("https://api.tavily.com/search", {
      method: "POST",
      headers: { "content-type": "application/json", authorization: `Bearer ${process.env.TAVILY_API_KEY}` },
      body: JSON.stringify({ query, max_results: 3 }),
    });
    if (!r.ok) throw new Error(`web search failed (${r.status})`);
    const j = await r.json();
    const results = j.results ?? [];
    return {
      engine: "Tavily",
      text: results.map((x) => `[${x.title}](${x.url})\n${x.content}`).join("\n\n---\n\n") || "No web results.",
      sources: results.map((x) => ({ title: x.title, url: x.url })),
    };
  }
  const res = await ai.models.generateContent({
    model: MODEL,
    contents: `Search the web for: ${query}\n\nReport what current sources say, with dates and names where they matter. Be factual and brief.`,
    config: { tools: [{ googleSearch: {} }], temperature: 0, maxOutputTokens: 1500, httpOptions: { retryOptions: RETRY } },
  });
  const chunks = res.candidates?.[0]?.groundingMetadata?.groundingChunks ?? [];
  const sources = chunks.map((c) => c.web).filter(Boolean).slice(0, 5).map((w) => ({ title: w.title || w.domain || "source", url: w.uri }));
  return {
    engine: "Google Search via Gemini",
    text: `${res.text ?? "No web results."}\n\nSources: ${sources.map((s) => s.title).join(", ") || "none listed"}`,
    sources,
  };
}

function finalText(parts) {
  return parts.filter((p) => p.text && !p.thought).map((p) => p.text).join("").trim();
}

function classify(err) {
  if (err instanceof ApiError) {
    if (err.status === 401 || err.status === 403 || (err.status === 400 && /api key/i.test(err.message))) return [502, "The server's Gemini key was rejected."];
    if (err.status === 429) return [429, "The Gemini quota for this demo is used up for now. Try again later."];
    if (err.status >= 500) return [503, "Gemini is busy right now. Try again in a moment."];
  }
  return [500, "The agent hit an error. Try again."];
}

export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).json({ error: "POST only" });
  const ip = (req.headers["x-forwarded-for"] || "").split(",")[0].trim() || req.socket?.remoteAddress || "?";
  if (limited(ip)) return res.status(429).json({ error: "Too many requests. Wait a minute and try again." });
  let body;
  try {
    body = typeof req.body === "string" ? JSON.parse(req.body || "{}") : req.body || {};
  } catch {
    return res.status(400).json({ error: "The request was not valid JSON." });
  }
  // Excerpts for passages the browser found, for when the agent itself is unavailable.
  if (Array.isArray(body.excerpts)) return res.status(200).json({ sources: papersResult(body.excerpts).sources });
  if (!ai) return res.status(503).json({ error: "The live agent is offline: no Gemini key is set on the server.", code: "no_key" });

  if (JSON.stringify(body).length > MAX_BODY) return res.status(413).json({ error: "The conversation is too long. Start a new one." });
  const contents = Array.isArray(body.contents) ? body.contents.filter((c) => c && (c.role === "user" || c.role === "model") && Array.isArray(c.parts)) : [];
  if (contents.length > 40) return res.status(413).json({ error: "The conversation is too long. Start a new one." });
  const trace = [];

  if (typeof body.question === "string") {
    const q = body.question.trim();
    if (!q || q.length > MAX_QUESTION) return res.status(400).json({ error: `Questions need to be between 1 and ${MAX_QUESTION} characters.` });
    contents.push({ role: "user", parts: [{ text: q }] });
  } else if (Array.isArray(body.retrieved)) {
    // The browser's answer to search_papers calls, plus any web results already run.
    const parts = Array.isArray(body.partial) ? body.partial.filter((p) => p?.functionResponse?.name === "search_web") : [];
    for (const r of body.retrieved.slice(0, 4)) {
      const { text, sources } = papersResult(Array.isArray(r.ids) ? r.ids : []);
      parts.push({ functionResponse: { id: r.id, name: "search_papers", response: { result: text } } });
      trace.push({ tool: "search_papers", query: String(r.query ?? "").slice(0, 300), sources });
    }
    if (!parts.length) return res.status(400).json({ error: "Nothing to continue." });
    contents.push({ role: "user", parts });
  } else {
    return res.status(400).json({ error: "Send a question." });
  }

  try {
    for (let round = 0; round < MAX_ROUNDS; round++) {
      const out = await ai.models.generateContent({
        model: MODEL,
        contents,
        config: { systemInstruction: SYSTEM_PROMPT, tools: TOOLS, temperature: 0, maxOutputTokens: 4096, httpOptions: { retryOptions: RETRY } },
      });
      const parts = out.candidates?.[0]?.content?.parts ?? [];
      if (!parts.length) return res.status(200).json({ type: "answer", text: "I could not produce an answer to that. Try rephrasing it.", contents, trace });
      // Model turns go back exactly as received: they carry the thought signatures Gemini needs.
      contents.push({ role: "model", parts });
      const calls = parts.filter((p) => p.functionCall).map((p) => p.functionCall);
      if (!calls.length) return res.status(200).json({ type: "answer", text: finalText(parts), contents, trace, model: MODEL });

      const partial = [], queries = [];
      for (const call of calls) {
        const query = String(call.args?.query ?? "").slice(0, 300);
        if (call.name === "search_web") {
          try {
            const w = await searchWeb(query);
            partial.push({ functionResponse: { id: call.id, name: "search_web", response: { result: w.text } } });
            trace.push({ tool: "search_web", query, engine: w.engine, sources: w.sources });
          } catch (err) {
            partial.push({ functionResponse: { id: call.id, name: "search_web", response: { error: "Web search is unavailable right now." } } });
            trace.push({ tool: "search_web", query, error: true, sources: [] });
          }
        } else if (call.name === "search_papers") {
          queries.push({ id: call.id, query });
        } else {
          partial.push({ functionResponse: { id: call.id, name: call.name, response: { error: `Unknown tool ${call.name}` } } });
        }
      }
      if (queries.length) return res.status(200).json({ type: "retrieve", queries, partial, contents, trace });
      contents.push({ role: "user", parts: partial });
    }
    return res.status(200).json({ type: "answer", text: "I searched several times without settling on an answer. Try a narrower question.", contents, trace });
  } catch (err) {
    console.error("agent error", err?.status, err?.message);
    const [status, message] = classify(err);
    return res.status(status).json({ error: message, trace });
  }
}
