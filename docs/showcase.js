/* Static showcase: reads docs/data/showcase.json and renders cached results. */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = (s) => String(s == null ? "" : s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
const attr = (s) => esc(s).replace(/"/g, "&quot;");
const num = (n) => (n == null ? "" : Number(n).toLocaleString("en-US"));
const pct = (n) => `${Math.round(n)}%`;
const sentimentClass = (s) => (s === "positive" ? "positive" : s === "negative" ? "negative" : "neutral");

let D = { hotel: null, reviews: [], mentions: [], taxonomy: {} };
let BY = {};
let CATS = [];
let active = null;

function categoryOrder() {
  const fromTax = Object.keys(D.taxonomy || {});
  const fromData = (D.hotel?.topics || []).map((t) => t.topic);
  const seen = [];
  for (const c of [...fromTax, ...fromData]) if (!seen.includes(c)) seen.push(c);
  return seen;
}
function topics() {
  const by = Object.fromEntries((D.hotel?.topics || []).map((t) => [t.topic, t]));
  return categoryOrder().filter((c) => by[c]).map((c) => by[c]);
}
function summary() {
  const h = D.hotel || {};
  const mt = h.mention_totals || {};
  const totals = h.totals || {};
  const mentions = (mt.negative || 0) + (mt.neutral || 0) + (mt.positive || 0);
  return { reviews: (D.reviews || []).length, avg: totals.avg != null ? Number(totals.avg).toFixed(2) : "-", mt, mentions };
}

function renderSummary() {
  const s = summary();
  $("#summaryReviews").textContent = `${num(s.reviews)} ${s.reviews === 1 ? "review" : "reviews"}`;
  $("#summaryDate").textContent = D.generated || "";
  $("#summaryMentions").textContent = `${num(s.mentions)} classified ${s.mentions === 1 ? "mention" : "mentions"}`;
  const cells = [
    ["Average rating", s.avg],
    ["Categories", num(topics().length)],
  ];
  $("#summaryMetrics").innerHTML = cells.map(([label, value]) => `<div class="metric"><span class="eyebrow">${esc(label)}</span><span class="metric-value">${esc(value)}</span></div>`).join("");
  const mt = D.hotel?.mention_totals || {};
  const rows = [["Negative mentions", mt.negative || 0, "negative"], ["Neutral mentions", mt.neutral || 0, "neutral"], ["Positive mentions", mt.positive || 0, "positive"]];
  const total = rows.reduce((a, r) => a + r[1], 0) || 1;
  $("#summaryLedger").innerHTML = rows.map(([label, n, kind]) => `<div class="ledger-row">
    <span class="ledger-label">${label}</span>
    <span class="ledger-value">${num(n)}</span>
    <span class="ledger-note">${pct((n / total) * 100)}</span>
    <span class="track"><i class="${kind}" style="width:${(n / total) * 100}%"></i></span>
  </div>`).join("");
}

function renderList() {
  const rows = (D.reviews || []).slice().sort((a, b) => (a.date < b.date ? 1 : -1));
  if (!rows.length) {
    $("#reviewList").innerHTML = `<p class="empty">No reviews available.</p>`;
    return;
  }
  $("#reviewList").innerHTML = rows.map((r) => {
    const idx = String((D.reviews || []).findIndex((x) => x.id === r.id) + 1).padStart(2, "0");
    return `<button class="review-item${r.id === active ? " active" : ""}" type="button" role="tab" aria-selected="${r.id === active}" aria-controls="detail" data-id="${attr(r.id)}">
      <span class="row"><span class="index">${idx}</span><span class="rating">${r.rating ? r.rating + " stars" : "no rating"}</span></span>
      <span class="preview">${esc(r.text)}</span>
      <span class="row"><span class="rid">${esc(r.date || "")}</span></span>
    </button>`;
  }).join("");
  $$("#reviewList .review-item").forEach((b) => b.addEventListener("click", () => {
    active = b.dataset.id;
    renderList();
    renderDetail();
    if (window.matchMedia("(max-width: 1000px)").matches) {
      $("#detail").scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }));
}

function renderDetail() {
  const review = (D.reviews || []).find((r) => r.id === active);
  if (!review) {
    $("#detail").innerHTML = `<p class="empty">Select a review.</p>`;
    return;
  }
  const mentions = (BY[review.id] || []).slice().sort((a, b) => String(a[0]).localeCompare(String(b[0])) || String(a[1]).localeCompare(String(b[1])));
  const cats = [...new Set(mentions.map((m) => m[0]))];
  const idx = String((D.reviews || []).findIndex((x) => x.id === review.id) + 1).padStart(2, "0");
  const rows = mentions.map((m) => `<tr>
    <td class="sentence">${esc(m[6] || "")}</td>
    <td class="cat">${esc(m[0])}</td>
    <td class="sentiment-${sentimentClass(m[1])}">${esc(m[1] || "neutral")}</td>
    <td class="mention-facet">${esc(m[2] || "\u2014")}</td>
  </tr>`).join("");
  $("#detail").innerHTML = `<div class="detail-content">
    <div class="detail-head">
      <div><p class="eyebrow">Review ${idx}</p>
        <h2 class="heading">${review.rating ? review.rating + " stars" : "No rating"} <span class="caption">&middot; ${esc(review.date || "")}</span></h2></div>

    </div>
    <div class="detail-meta">${cats.map((c) => `<span class="chip">${esc(c)}</span>`).join("") || `<span class="chip neutral">No category</span>`}</div>
    <p class="detail-text">${esc(review.text)}</p>
    <div class="detail-section">
      <p class="eyebrow">Mentions</p>
      ${mentions.length ? `<table class="mention-table"><thead><tr><th>Sentence</th><th>Category</th><th>Sentiment</th><th>Facet</th></tr></thead><tbody>${rows}</tbody></table>` : `<p class="empty">No mentions in this review.</p>`}
    </div>
  </div>`;
}

function render() {
  const rows = (D.reviews || []).slice().sort((a, b) => (a.date < b.date ? 1 : -1));
  if (!active || !rows.some((r) => r.id === active)) active = rows[0]?.id || null;
  renderSummary();
  renderList();
  renderDetail();
}

function wireMotion() {
  const progress = $("#progress");
  const update = () => {
    const max = document.documentElement.scrollHeight - window.innerHeight;
    progress.style.width = `${max > 0 ? (window.scrollY / max) * 100 : 0}%`;
  };
  addEventListener("scroll", update, { passive: true });
  update();
  if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    $$(".rule").forEach((r) => r.classList.add("draw"));
  }
}

async function boot() {
  wireMotion();
  try {
    const resp = await fetch("data/showcase.json", { cache: "no-store" });
    if (!resp.ok) throw new Error(`showcase.json returned ${resp.status}`);
    D = await resp.json();
    for (const m of D.mentions || []) {
      const rid = m[7];
      (BY[rid] || (BY[rid] = [])).push(m);
    }
    CATS = categoryOrder();
    render();
  } catch (e) {
    $("#reviewList").innerHTML = `<p class="empty">Could not load the cached results.</p>`;
    $("#detail").innerHTML = `<p class="empty">${esc(e.message || e)}</p>`;
  }
}

boot();
