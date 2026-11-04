/* Aspect-Based Hotel Review Analysis: one property, local batch import, overview with facets. */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = (s) => String(s == null ? "" : s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
const attr = (s) => esc(s).replace(/"/g, "&quot;");
const num = (n) => (n == null ? "" : Number(n).toLocaleString("en-US"));
const pct = (n) => `${Math.round(n)}%`;

let D = { hotel: null, reviews: [], mentions: [], taxonomy: {} };
let CATS = [];
const state = { importing: false, open: "", played: false };


function categoryOrder() {
  const fromTax = Object.keys(D.taxonomy || {});
  const fromData = (D.hotel?.topics || []).map((t) => t.topic);
  const seen = [];
  for (const c of [...fromTax, ...fromData]) if (!seen.includes(c)) seen.push(c);
  return seen.length ? seen : CATS;
}

function topics() {
  const rows = (D.hotel?.topics || []).slice();
  const by = Object.fromEntries(rows.map((t) => [t.topic, t]));
  return categoryOrder().filter((c) => by[c]).map((c) => by[c]);
}

function summary() {
  const h = D.hotel;
  const totals = h?.totals || {};
  const mt = h?.mention_totals || {};
  const reviews = D.reviews || [];
  const rated = reviews.map((r) => r.rating).filter((r) => r);
  const avg = totals.avg != null ? Number(totals.avg).toFixed(2) : (rated.length ? (rated.reduce((a, b) => a + b, 0) / rated.length).toFixed(2) : "-");
  const mentions = (mt.negative || 0) + (mt.neutral || 0) + (mt.positive || 0);
  return { totals, mt, avg, mentions, reviews: reviews.length };
}

function renderMetrics() {
  const s = summary();
  $("#summaryReviews").textContent = `${num(s.reviews)} ${s.reviews === 1 ? "review" : "reviews"}`;
  $("#summaryDate").textContent = D.generated || "";
  $("#summaryMentions").textContent = `${num(s.mentions)} classified ${s.mentions === 1 ? "mention" : "mentions"}`;
  const cells = [
    ["Average rating", s.avg],
    ["Categories", num(topics().length)],
  ];
  $("#summaryMetrics").innerHTML = cells.map(([label, value]) => `<div class="metric"><span class="eyebrow">${esc(label)}</span><span class="metric-value">${esc(value)}</span></div>`).join("");
}

function renderLedger() {
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

function renderCategories() {
  $("#categoryGrid").innerHTML = topics().map((t) => {
    const n = t.neg + t.pos + t.neutral;
    const rows = [["Negative", t.neg, "negative"], ["Neutral", t.neutral, "neutral"], ["Positive", t.pos, "positive"]];
    return `<div class="category-card">
      <div class="top"><span class="name">${esc(t.topic)}</span><span class="count">${num(n)} mentions</span></div>
      <div class="sent-rows">${rows.map(([label, count, kind]) => `<div class="sent-row ${kind}"><span>${label}</span><span class="num">${num(count)}</span><span class="bar"><i style="width:${n ? (count / n) * 100 : 0}%"></i></span></div>`).join("")}</div>
    </div>`;
  }).join("");
}

function facetRows(topic) {
  const rows = [];
  for (const [kind, list] of [["negative", topic.complaints || []], ["positive", topic.praise || []]]) {
    for (const f of list) {
      if (!f || !f.theme) continue;
      const q = (f.quotes || [])[0];
      rows.push({ theme: f.theme, count: f.count || 0, kind, quote: q, vague: f.theme === "General Experience" });
    }
  }
  return rows.sort((a, b) => b.count - a.count);
}

function facetHtml(f) {
  return `<div class="facet">
    <span class="facet-name">${esc(f.theme)}</span>
    <div>${f.quote ? `<p class="facet-quote quote" data-id="${attr(f.quote.id || "")}" data-ev="${attr(f.quote.text || "")}">&ldquo;${esc(f.quote.text)}&rdquo;<span class="qmeta">${[f.quote.rating ? f.quote.rating + " stars" : "", f.quote.date].filter(Boolean).map(esc).join(" &middot; ")}</span></p>` : `<p class="facet-quote muted">No quote for this facet.</p>`}</div>
    <span class="facet-count"><span class="chip ${f.kind}">${f.kind === "negative" ? "complaint" : "praise"}</span><br>${num(f.count)}</span>
  </div>`;
}

function renderFacets() {
  const list = topics().slice().sort((a, b) => (b.neg + b.pos + b.neutral) - (a.neg + a.pos + a.neutral));
  const open = state.open || list[0]?.topic || "";
  state.open = open;
  $("#facetAccordion").innerHTML = list.map((t) => {
    const n = t.neg + t.pos + t.neutral;
    const allFacets = facetRows(t);
    const vague = allFacets.filter((f) => f.vague);
    const facets = allFacets.filter((f) => !f.vague).slice(0, 6);
    return `<div class="acc-item${t.topic === open ? " open" : ""}" data-topic="${attr(t.topic)}">
      <button class="acc-head" type="button" aria-expanded="${t.topic === open}">
        <span class="acc-title">${esc(t.topic)}</span>
        <span class="acc-meta">${num(n)} mentions &middot; ${pct((t.neg / (n || 1)) * 100)} negative</span>
        <span class="chev">+</span>
      </button>
      <div class="acc-body"><div><div class="acc-inner">
        ${facets.length ? `<div class="facet-list">${facets.map(facetHtml).join("")}</div>` : `<p class="empty">No specific facets in this batch.</p>`}
        ${vague.length ? `<div class="vague-block"><p class="eyebrow">General experience</p><p class="vague-note">Catch-all mentions that do not name a specific facet.</p><div class="facet-list">${vague.map(facetHtml).join("")}</div></div>` : ""}
      </div></div></div>
    </div>`;
  }).join("");
  $$("#facetAccordion .acc-head").forEach((b) => b.addEventListener("click", () => {
    const item = b.closest(".acc-item");
    const topic = item.dataset.topic;
    state.open = state.open === topic ? "" : topic;
    renderFacets();
  }));
}

function renderDashboard() {
  if (state.played) {
    const hero = $("#dashboardView .hero");
    if (hero) hero.classList.remove("set");
  }
  const s = summary();
  const h = D.hotel;
  const weakest = h?.verdict?.topic || "";
  const detail = h?.verdict?.detail || "";
  $("#dashEyebrow").textContent = `Single property \u00b7 ${D.generated || "overview"}`;
  $("#dashLine1").textContent = weakest ? `${weakest}` : "What guests";
  $("#dashLine2").textContent = weakest ? "needs attention." : "talk about.";
  $("#dashLede").textContent = weakest
    ? `${num(s.reviews)} reviews \u00b7 ${num(s.mentions)} classified mentions. ${detail || `${weakest} has the highest negative share in this batch.`}`
    : `${num(s.reviews)} reviews \u00b7 ${num(s.mentions)} classified mentions in this batch. Import another CSV to replace it.`;
  renderMetrics();
  renderLedger();
  renderCategories();
  renderFacets();
  state.played = true;
}

function showView(which) {
  $("#importView").hidden = which !== "import";
  $("#dashboardView").hidden = which !== "dashboard";
}

function render() {
  if (D.hotel) {
    showView("dashboard");
    renderDashboard();
  } else {
    showView("import");
  }
  animateReveals();
}

function animateReveals() {
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  $$(".rule").forEach((r) => r.classList.add("draw"));
  const io = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting) {
        e.target.classList.add("reveal");
        io.unobserve(e.target);
      }
    });
  }, { rootMargin: "-10% 0px" });
  $$(".category-grid, .accordion, #summaryPanel").forEach((el) => io.observe(el));
}

function toast(message, isError = false) {
  const el = $("#toast");
  el.textContent = message;
  el.classList.toggle("error", isError);
  el.classList.add("show");
  clearTimeout(el._timer);
  el._timer = setTimeout(() => el.classList.remove("show"), 5000);
}

function setImportStatus(message, kind = "") {
  const el = $("#importStatus");
  el.textContent = message;
  el.className = `status-line ${kind}`;
}

async function importFile(file) {
  if (!file || state.importing) return;
  if (!/\.csv$/i.test(file.name) && file.type !== "text/csv") {
    setImportStatus("Choose a .csv file.", "error");
    return;
  }
  state.importing = true;
  $("#importBtn").disabled = true;
  setImportStatus(`Processing ${file.name}. The first run can take a minute while the models load.`);
  try {
    const resp = await fetch("/api/import", {
      method: "POST",
      headers: { "Content-Type": "text/csv", "X-Filename": encodeURIComponent(file.name) },
      body: file,
    });
    const body = await resp.json().catch(() => ({}));
    if (!resp.ok || !body.ok) throw new Error(body.error || `Import failed (${resp.status})`);
    setImportStatus(`Imported ${num(body.reviews)} reviews and ${num(body.mentions)} mentions.`, "ok");
    await loadData();
    toast(`Imported ${num(body.reviews)} reviews.`);
  } catch (e) {
    showView("import");
    setImportStatus(String(e.message || e), "error");
    toast(String(e.message || e), true);
  } finally {
    state.importing = false;
    $("#importBtn").disabled = false;
  }
}

async function loadData() {
  const resp = await fetch("/api/data", { cache: "no-store" });
  D = await resp.json();
  CATS = categoryOrder();
  render();
}

function wireImport() {
  const input = $("#fileInput");
  const input2 = $("#fileInput2");
  const drop = $("#drop");
  $("#importBtn").addEventListener("click", () => input2.click());
  input.addEventListener("change", () => importFile(input.files[0]));
  input2.addEventListener("change", () => importFile(input2.files[0]));
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => {
    e.preventDefault();
    drop.classList.add("over");
  }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => {
    e.preventDefault();
    drop.classList.remove("over");
  }));
  drop.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files[0];
    if (f) importFile(f);
  });
}

function wireMotion() {
  const progress = $("#progress");
  const update = () => {
    const max = document.documentElement.scrollHeight - window.innerHeight;
    progress.style.width = `${max > 0 ? (window.scrollY / max) * 100 : 0}%`;
  };
  addEventListener("scroll", update, { passive: true });
  update();
}

function openReviewModal(id, ev) {
  const review = (D.reviews || []).find((r) => String(r.id) === String(id));
  const modal = $("#modal");
  if (!review || !modal) return;
  const meta = [review.rating ? review.rating + " stars" : "", review.date].filter(Boolean).join(" \u00b7 ");
  modal.innerHTML = `<div class="modal-backdrop" style="position:fixed;inset:0;background:rgba(12,12,11,.78);display:grid;place-items:center;padding:24px;z-index:80">
    <div class="modal-card panel" style="max-width:760px;max-height:80vh;overflow:auto;box-shadow:0 1px 3px rgba(0,0,0,.24),0 8px 24px rgba(0,0,0,.32)">
      <div class="panel-head"><p class="eyebrow">Review</p><button class="btn ghost compact" type="button" data-modal-close>Close</button></div>
      <p class="caption">${esc(meta)}</p>
      <p class="detail-text" style="border:0;padding:0;margin-top:16px">${esc(review.text || "")}</p>
      ${ev ? `<p class="facet-quote" style="margin-top:16px">&ldquo;${esc(ev)}&rdquo;</p>` : ""}
    </div>
  </div>`;
  modal.hidden = false;
}
function closeModal() { const m = $("#modal"); if (m) { m.hidden = true; m.innerHTML = ""; } }

async function boot() {
  wireImport();
  wireMotion();
  document.addEventListener("click", (e) => {
    if (e.target.closest("[data-modal-close]") || (e.target.classList && e.target.classList.contains("modal-backdrop"))) { closeModal(); return; }
    const q = e.target.closest(".quote");
    if (q) openReviewModal(q.dataset.id, q.dataset.ev);
  });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeModal(); });
  try {
    await loadData();
  } catch (e) {
    showView("import");
    setImportStatus("Could not read /api/data. Start the local server with run.bat.", "error");
  }
}

boot();
