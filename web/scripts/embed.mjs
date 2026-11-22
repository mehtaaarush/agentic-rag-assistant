// Embed the corpus chunks with all-MiniLM-L6-v2 (the model src/ingest.py uses), in the same
// 8-bit ONNX build the browser loads, so stored and query vectors come from identical weights.
//
//   npm install --no-save @huggingface/transformers@3.8.1
//   node scripts/embed.mjs
//
// Writes public/data/embeddings.bin (int8 rows, then one float32 scale per row) and index.json.
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { pipeline } from "@huggingface/transformers";

const WEB = join(dirname(fileURLToPath(import.meta.url)), "..");
const MODEL = "Xenova/all-MiniLM-L6-v2";
const { chunks } = JSON.parse(readFileSync(join(WEB, "lib", "corpus.json"), "utf8"));

const extractor = await pipeline("feature-extraction", MODEL, { dtype: "q8" });
const dim = 384, n = chunks.length;
const q = new Int8Array(n * dim), scale = new Float32Array(n);
const BATCH = 32;
for (let s = 0; s < n; s += BATCH) {
  const texts = chunks.slice(s, s + BATCH).map((c) => c.text);
  const out = await extractor(texts, { pooling: "mean", normalize: true });
  for (let r = 0; r < texts.length; r++) {
    const row = out.data.subarray(r * dim, (r + 1) * dim);
    let m = 0;
    for (const v of row) m = Math.max(m, Math.abs(v));
    scale[s + r] = m / 127;
    for (let k = 0; k < dim; k++) q[(s + r) * dim + k] = Math.round(row[k] / scale[s + r]);
  }
  process.stdout.write(`\r${Math.min(s + BATCH, n)} / ${n}`);
}
process.stdout.write("\n");
const buf = Buffer.concat([Buffer.from(q.buffer), Buffer.from(scale.buffer)]);
writeFileSync(join(WEB, "public", "data", "embeddings.bin"), buf);
writeFileSync(
  join(WEB, "public", "data", "index.json"),
  JSON.stringify({ model: MODEL, dtype: "q8", dim, count: n, paper: chunks.map((c) => c.paper), page: chunks.map((c) => c.page) }),
);
console.log(`embeddings.bin ${(buf.length / 1e6).toFixed(2)} MB`);
