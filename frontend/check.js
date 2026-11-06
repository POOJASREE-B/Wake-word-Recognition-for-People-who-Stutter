/* Live check: everything on this tab is computed from the REAL recordings (stutterRealPeople -> real_people/) with the model
   that is loaded right now, from the stored leave-speakers-out cross-validation, from the false-wake-up replay (farh_report.json)
   and from the acoustic pattern analysis of the recordings. Nothing is a mock-up. */
const PH_ORDER = ["hey_google", "hey_siri", "hey_alexa", "alexa"];
const PH_LABEL = { hey_google: "Hey Google (wake word)", hey_siri: "Hey Siri", hey_alexa: "Hey Alexa", alexa: "Alexa" };
function mkCanvas(card, w, h) {
  const cv = el("canvas"), r = window.devicePixelRatio || 1;
  cv.width = w * r; cv.height = h * r; cv.style.width = "100%"; cv.style.maxWidth = w + "px"; cv.style.height = "auto";
  const c = cv.getContext("2d"); c.scale(r, r); c.font = "12px " + (getComputedStyle(document.body).fontFamily || "sans-serif");
  card.appendChild(cv); return { cv, c, w, h };
}
function axisX(c, x0, x1, y, v0, v1, ticks, fmt) {
  c.strokeStyle = cssVar("--line"); c.fillStyle = cssVar("--ink-muted"); c.textAlign = "center"; c.lineWidth = 1;
  c.beginPath(); c.moveTo(x0, y); c.lineTo(x1, y); c.stroke();
  ticks.forEach(t => { const x = x0 + (t - v0) / (v1 - v0) * (x1 - x0); c.beginPath(); c.moveTo(x, y); c.lineTo(x, y + 4); c.stroke(); c.fillText(fmt ? fmt(t) : String(t), x, y + 17); });
}
function legend(c, items, x, y) {
  c.textAlign = "left";
  items.forEach(([label, color, kind]) => {
    c.strokeStyle = color; c.fillStyle = color; c.lineWidth = 2;
    if (kind === "ring") { c.beginPath(); c.arc(x + 5, y - 4, 4, 0, 6.3); c.stroke(); }
    else if (kind === "dot") { c.beginPath(); c.arc(x + 5, y - 4, 4, 0, 6.3); c.fill(); }
    else if (kind === "dash") { c.setLineDash([4, 3]); c.beginPath(); c.moveTo(x, y - 4); c.lineTo(x + 14, y - 4); c.stroke(); c.setLineDash([]); }
    else { c.beginPath(); c.moveTo(x, y - 4); c.lineTo(x + 14, y - 4); c.stroke(); }
    c.fillStyle = cssVar("--ink"); c.fillText(label, x + 20, y); x += 26 + c.measureText(label).width;
  });
}
const rate = (arr, f) => arr.length ? arr.filter(f).length / arr.length : null;

function chartScores(card, live, held, thr, soft) {
  const { c, w, h } = mkCanvas(card, 760, 250), L = 150, R = 20, T = 34, B = 34, rowH = (h - T - B) / PH_ORDER.length;
  const X = v => L + v * (w - L - R);
  c.fillStyle = cssVar("--listening"); c.globalAlpha = 0.14; c.fillRect(X(Math.max(0, thr - soft)), T, X(thr) - X(Math.max(0, thr - soft)), h - T - B); c.globalAlpha = 1;
  PH_ORDER.forEach((p, i) => {
    const y0 = T + i * rowH; c.fillStyle = cssVar("--ink"); c.textAlign = "right"; c.fillText(PH_LABEL[p], L - 10, y0 + rowH / 2 + 4);
    c.strokeStyle = cssVar("--line"); c.beginPath(); c.moveTo(L, y0 + rowH); c.lineTo(w - R, y0 + rowH); c.stroke();
    const jit = k => ((k * 37) % 11) / 11 - 0.5;
    live.filter(x => x.phrase === p).forEach((x, k) => {
      c.fillStyle = p === "hey_google" ? (x.score >= thr ? cssVar("--accepted") : cssVar("--disfluent")) : cssVar("--ink-muted");
      c.beginPath(); c.arc(X(x.score), y0 + rowH / 2 + jit(k) * (rowH - 12) - 3, 4, 0, 6.3); c.fill();
    });
    held.filter(x => x.phrase === p).forEach((x, k) => {
      c.strokeStyle = cssVar("--fluent"); c.lineWidth = 1.6;
      c.beginPath(); c.arc(X(Math.min(1, x.score)), y0 + rowH / 2 + jit(k + 5) * (rowH - 12) + 4, 4, 0, 6.3); c.stroke();
    });
  });
  c.strokeStyle = cssVar("--ink"); c.lineWidth = 1.5; c.setLineDash([5, 3]); c.beginPath(); c.moveTo(X(thr), T - 6); c.lineTo(X(thr), h - B); c.stroke(); c.setLineDash([]);
  c.fillStyle = cssVar("--ink"); c.textAlign = "center"; c.fillText("threshold " + thr.toFixed(2), X(thr), T - 10);
  axisX(c, L, w - R, h - B, 0, 1, [0, 0.2, 0.4, 0.6, 0.8, 1], v => v.toFixed(1)); c.textAlign = "center"; c.fillStyle = cssVar("--ink-muted"); c.fillText("score", (L + w - R) / 2, h - 4);
  legend(c, [["accepted (in-sample)", cssVar("--accepted"), "dot"], ["missed (in-sample)", cssVar("--disfluent"), "dot"], ["held-out speaker", cssVar("--fluent"), "ring"]], L, 14);
}
function chartCurves(card, sweep, held, thr) {
  const { c, w, h } = mkCanvas(card, 760, 280), L = 56, R = 20, T = 36, B = 38;
  const X = v => L + v * (w - L - R), Y = v => T + (1 - v) * (h - T - B);
  c.strokeStyle = cssVar("--line"); c.fillStyle = cssVar("--ink-muted"); c.textAlign = "right";
  [0, 0.25, 0.5, 0.75, 1].forEach(v => { c.beginPath(); c.moveTo(L, Y(v)); c.lineTo(w - R, Y(v)); c.stroke(); c.fillText(Math.round(v * 100) + "%", L - 6, Y(v) + 4); });
  const hp = held.filter(x => x.wake).map(x => x.score), hn = held.filter(x => !x.wake).map(x => x.score);
  const line = (pts, color, dash) => { c.strokeStyle = color; c.lineWidth = 2; c.setLineDash(dash || []); c.beginPath(); pts.forEach(([x, y], i) => i ? c.lineTo(X(x), Y(y)) : c.moveTo(X(x), Y(y))); c.stroke(); c.setLineDash([]); };
  const ts = sweep.map(s => s.t);
  line(sweep.map(s => [s.t, s.recall]), cssVar("--accepted"));
  line(ts.map(t => [t, rate(hp, v => v >= t)]), cssVar("--accepted"), [5, 3]);
  line(sweep.map(s => [s.t, s.far]), cssVar("--disfluent"));
  line(ts.map(t => [t, rate(hn, v => v >= t)]), cssVar("--disfluent"), [5, 3]);
  c.strokeStyle = cssVar("--ink"); c.lineWidth = 1.5; c.setLineDash([5, 3]); c.beginPath(); c.moveTo(X(thr), T - 4); c.lineTo(X(thr), h - B); c.stroke(); c.setLineDash([]);
  c.fillStyle = cssVar("--ink"); c.textAlign = "center"; c.fillText("now " + thr.toFixed(2), X(thr), T - 8);
  axisX(c, L, w - R, h - B, 0, 1, [0, 0.2, 0.4, 0.6, 0.8, 1], v => v.toFixed(1)); c.fillStyle = cssVar("--ink-muted"); c.fillText("threshold", (L + w - R) / 2, h - 4);
  legend(c, [["Hey Google accepted, in-sample", cssVar("--accepted"), "line"], ["held-out", cssVar("--accepted"), "dash"], ["other phrases accepted, in-sample", cssVar("--disfluent"), "line"], ["held-out", cssVar("--disfluent"), "dash"]], 8, 14);
}
function chartFarh(card, farh, held, thr) {
  const { c, w, h } = mkCanvas(card, 760, 290), L = 62, R = 62, T = 36, B = 38, lo = Math.log10(0.1), hi = Math.log10(200);
  const X = v => L + v * (w - L - R), YL = v => T + (1 - (Math.log10(Math.max(v, 0.1)) - lo) / (hi - lo)) * (h - T - B), YR = v => T + (1 - v) * (h - T - B);
  c.strokeStyle = cssVar("--line"); c.fillStyle = cssVar("--ink-muted");
  [0.1, 1, 10, 100].forEach(v => { c.textAlign = "right"; c.beginPath(); c.moveTo(L, YL(v)); c.lineTo(w - R, YL(v)); c.stroke(); c.fillText(v + "/h", L - 6, YL(v) + 4); });
  [0, 0.25, 0.5, 0.75, 1].forEach(v => { c.textAlign = "left"; c.fillText(Math.round(v * 100) + "%", w - R + 8, YR(v) + 4); });
  const g = (farh.far_grid || []).filter(r => r.threshold >= 0.3);
  c.fillStyle = cssVar("--disfluent"); c.globalAlpha = 0.16; c.beginPath();
  g.forEach((r, i) => { const x = X(r.threshold), y = YL(r.pooled.ci95[1]); i ? c.lineTo(x, y) : c.moveTo(x, y); });
  [...g].reverse().forEach(r => c.lineTo(X(r.threshold), YL(r.pooled.ci95[0]))); c.closePath(); c.fill(); c.globalAlpha = 1;
  c.strokeStyle = cssVar("--disfluent"); c.lineWidth = 2; c.beginPath(); g.forEach((r, i) => { const x = X(r.threshold), y = YL(r.pooled.per_hour); i ? c.lineTo(x, y) : c.moveTo(x, y); }); c.stroke();
  const hp = held.filter(x => x.wake).map(x => x.score);
  c.strokeStyle = cssVar("--fluent"); c.setLineDash([5, 3]); c.beginPath();
  g.forEach((r, i) => { const x = X(r.threshold), y = YR(1 - rate(hp, v => v >= r.threshold)); i ? c.lineTo(x, y) : c.moveTo(x, y); }); c.stroke(); c.setLineDash([]);
  c.strokeStyle = cssVar("--ink"); c.lineWidth = 1.5; c.setLineDash([5, 3]); c.beginPath(); c.moveTo(X(thr), T - 4); c.lineTo(X(thr), h - B); c.stroke(); c.setLineDash([]);
  c.fillStyle = cssVar("--ink"); c.textAlign = "center"; c.fillText("now " + thr.toFixed(2), X(thr), T - 8);
  axisX(c, L, w - R, h - B, 0, 1, [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], v => v.toFixed(1));
  c.save(); c.translate(0, 0); c.restore();
  legend(c, [["false wake-ups per hour (" + (farh.negative_hours || 0).toFixed(1) + " h of negative audio, 95% band)", cssVar("--disfluent"), "line"], ["Hey Google missed, held-out (right axis)", cssVar("--fluent"), "dash"]], 8, 14);
  c.fillStyle = cssVar("--ink-muted"); c.textAlign = "center"; c.fillText("threshold (a false wake-up rate of 0 is drawn at the 0.1/h floor: it only means none in this pool)", (L + w - R) / 2, h - 4);
}
function chartPattern(card, pat) {
  const cats = [["long gap ≥0.25 s", p => p.blocks.length], ["pause 0.12-0.25 s", p => p.pauses.length], ["repetition", p => p.reps], ["prolongation", p => p.prolong.length], ["phrase restart", p => p.restarts]];
  const { c, w, h } = mkCanvas(card, 760, 270), L = 56, R = 16, T = 36, B = 50, gw = (w - L - R) / PH_ORDER.length;
  const colors = [cssVar("--disfluent"), cssVar("--listening"), cssVar("--fluent"), cssVar("--primary"), cssVar("--ink-muted")];
  const Y = v => T + (1 - v) * (h - T - B);
  c.strokeStyle = cssVar("--line"); c.fillStyle = cssVar("--ink-muted"); c.textAlign = "right";
  [0, 0.25, 0.5, 0.75, 1].forEach(v => { c.beginPath(); c.moveTo(L, Y(v)); c.lineTo(w - R, Y(v)); c.stroke(); c.fillText(Math.round(v * 100) + "%", L - 6, Y(v) + 4); });
  PH_ORDER.forEach((p, i) => {
    const rows = pat.filter(x => x.phrase === p), bw = (gw - 24) / cats.length;
    cats.forEach(([name, f], k) => {
      const v = rows.length ? rows.filter(x => f(x) > 0).length / rows.length : 0, x = L + i * gw + 12 + k * bw;
      c.fillStyle = colors[k]; c.fillRect(x, Y(v), bw - 3, Y(0) - Y(v));
      c.fillStyle = cssVar("--ink"); c.textAlign = "center"; c.fillText(rows.filter(r => f(r) > 0).length + "/" + rows.length, x + (bw - 3) / 2, Y(v) - 4);
    });
    c.fillStyle = cssVar("--ink"); c.textAlign = "center"; c.fillText(PH_LABEL[p].replace(" (wake word)", ""), L + i * gw + gw / 2, h - B + 18);
  });
  legend(c, cats.map(([n], k) => [n, colors[k], "dot"]), 8, 14);
}
function chartSegments(card, pat, phrase) {
  const rows = pat.filter(x => x.phrase === phrase && x.segs.length).sort((a, b) => (a.span || 0) - (b.span || 0));
  const rh = 15, { c, w, h } = mkCanvas(card, 760, 56 + rows.length * rh), L = 70, R = 16, T = 24, B = 28, tmax = Math.max(4, ...rows.map(r => r.segs[r.segs.length - 1][1]));
  const X = t => L + t / tmax * (w - L - R);
  rows.forEach((r, i) => {
    const y = T + i * rh; c.fillStyle = cssVar("--ink-muted"); c.textAlign = "right"; c.fillText(r.speaker, L - 8, y + 11);
    c.strokeStyle = cssVar("--line"); c.beginPath(); c.moveTo(L, y + rh / 2); c.lineTo(w - R, y + rh / 2); c.stroke();
    r.segs.forEach(([a, b]) => { c.fillStyle = cssVar("--primary"); c.fillRect(X(a), y + 2, Math.max(2, X(b) - X(a)), rh - 5); });
    for (let k = 0; k + 1 < r.segs.length; k++) { const g = r.segs[k + 1][0] - r.segs[k][1]; if (g >= 0.25) { c.fillStyle = cssVar("--disfluent"); c.globalAlpha = 0.5; c.fillRect(X(r.segs[k][1]), y + 2, X(r.segs[k + 1][0]) - X(r.segs[k][1]), rh - 5); c.globalAlpha = 1; } }
  });
  axisX(c, L, w - R, T + rows.length * rh + 2, 0, tmax, [0, 1, 2, 3, 4, 5, 6, 7, 8].filter(t => t <= tmax + 0.5), t => t + " s");
  legend(c, [["voiced stretch", cssVar("--primary"), "dot"], ["silent gap ≥ 0.25 s", cssVar("--disfluent"), "dot"]], L, 14);
}
function barsFromCounts(card, title, obj, color) {
  const ent = Object.entries(obj || {}).sort((a, b) => b[1] - a[1]).slice(0, 12), rh = 20, { c, w, h } = mkCanvas(card, 760, 28 + ent.length * rh), L = 190, R = 70, mx = Math.max(1, ...ent.map(e => e[1]));
  c.fillStyle = cssVar("--ink-muted"); c.textAlign = "left"; c.fillText(title, 0, 12);
  ent.forEach(([k, v], i) => { const y = 24 + i * rh; c.fillStyle = cssVar("--ink"); c.textAlign = "right"; c.fillText(k, L - 8, y + 12);
    c.fillStyle = color; c.fillRect(L, y + 2, (w - L - R) * v / mx, rh - 6); c.fillStyle = cssVar("--ink"); c.textAlign = "left"; c.fillText(String(v), L + (w - L - R) * v / mx + 6, y + 12); });
}

async function getJson(url, tries) {
  for (let i = 0; i < (tries || 1); i++) {
    const r = await fetch(url); const j = await r.json().catch(() => ({}));
    if (r.status === 202) { $("#checkMsg").textContent = "Scoring your recordings with the loaded model: " + j.done + " of " + j.total; await new Promise(res => setTimeout(res, 1200)); continue; }
    if (!r.ok) throw new Error(j.error || r.statusText);
    return j;
  }
  throw new Error("timed out");
}
async function loadCheck() {
  const box = $("#checkBox"); box.innerHTML = ""; $("#checkMsg").textContent = "Loading";
  let live, held = null, pat = null, ds = null, heldErr = "";
  try { live = await getJson("/api/live/eval", 600); }
  catch (e) { box.innerHTML = ""; const c = el("div", "card"); c.appendChild(el("p", "empty", "Could not score the real recordings: " + e.message)); box.appendChild(c); $("#checkMsg").textContent = ""; return; }
  try { held = await getJson("/api/live/heldout"); } catch (e) { heldErr = e.message; }
  try { pat = (await getJson("/api/live/patterns", 60)).clips; } catch (e) { /* shown below */ }
  try { ds = await getJson("/api/dataset/summary"); } catch (e) { /* optional */ }
  $("#checkMsg").textContent = ""; box.innerHTML = "";
  const m = live.model, thr = m.threshold, soft = m.soft_margin || 0, clips = live.clips, hc = held ? held.clips : [];
  const pos = clips.filter(x => x.wake), neg = clips.filter(x => !x.wake);
  const hit = pos.filter(x => x.score >= thr).length, fa = neg.filter(x => x.score >= thr).length;
  const hpos = hc.filter(x => x.wake), hneg = hc.filter(x => !x.wake);
  const hhit = hpos.filter(x => x.score >= thr).length, hfa = hneg.filter(x => x.score >= thr).length;

  const head = el("div", "card");
  head.appendChild(el("h2", "", "The model that is loaded now"));
  head.appendChild(el("p", "", "Type: " + m.type + ". Threshold: " + thr.toFixed(3) + " (the confirming check may dip " + soft.toFixed(2) + " below it). Model file written " + m.model_file_time + ", decision layer " + m.verifier_time + "."));
  head.appendChild(el("p", "hint", "In-sample means the model was trained on these recordings, so those numbers flatter it. Held-out means each speaker was scored by a model that never heard that speaker (cross-validation" + (held ? ", " + held.k + " folds, report written " + held.time : "") + "). The held-out numbers are the ones to quote."));
  box.appendChild(head);

  const grid = el("div", "metrics");
  grid.appendChild(metric("Hey Google accepted, in-sample", hit + " of " + pos.length, "Real recordings, loaded model, threshold " + thr.toFixed(2) + "."));
  grid.appendChild(metric("Other phrases accepted, in-sample", fa + " of " + neg.length, "Hey Siri, Alexa, Hey Alexa recordings."));
  if (held) {
    grid.appendChild(metric("Hey Google accepted, held-out", hhit + " of " + hpos.length, "Held-out scores at the same threshold."));
    grid.appendChild(metric("Other phrases accepted, held-out", hfa + " of " + hneg.length, "Held-out scores at the same threshold."));
  }
  grid.appendChild(metric("Separation (AUC), in-sample", live.auc == null ? "-" : live.auc.toFixed(3), "1.0 is perfect ranking; only 3 other phrases are tested."));
  if (held) grid.appendChild(metric("Separation (AUC), held-out", held.auc == null ? "-" : Number(held.auc).toFixed(3), "Wide uncertainty with " + hpos.length + " Hey Google clips."));
  box.appendChild(grid);

  let card = el("div", "card"); card.appendChild(el("h2", "", "Score of every real recording"));
  card.appendChild(el("p", "hint", "One dot per recording. Filled dots: scored by the loaded model (trained on them). Rings: scored by a model that never heard that speaker. Shaded band: where the confirming check is still accepted."));
  chartScores(card, clips, hc, thr, soft); box.appendChild(card);

  card = el("div", "card"); card.appendChild(el("h2", "", "Accepted share against threshold"));
  card.appendChild(el("p", "hint", "Green: share of Hey Google recordings accepted. Red: share of other-phrase recordings accepted. Solid is in-sample, dashed is held-out."));
  chartCurves(card, live.sweep, hc, thr); box.appendChild(card);

  card = el("div", "card"); card.appendChild(el("h2", "", "False wake-ups per hour against threshold"));
  if (live.farh) {
    if (live.farh_stale) card.appendChild(el("p", "msg-error", "This false-wake-up report (" + live.farh_time + ") was made for an earlier model than the one loaded now (" + m.model_file_time + ")."));
    else card.appendChild(el("p", "hint", "Report written " + live.farh_time + ". Replay of " + live.farh.negative_hours.toFixed(2) + " h of audio without the wake word through the live loop. Chosen on the same audio, so the rate at the chosen threshold is optimistic."));
    chartFarh(card, live.farh, hc, thr);
  } else { card.appendChild(el("p", "empty", "No false-wake-up report is available on this server.")); }
  box.appendChild(card);

  card = el("div", "card"); card.appendChild(el("h2", "", "How the people in the recordings stutter"));
  if (pat) {
    card.appendChild(el("p", "hint", "Share of recordings in which each pattern was measured (acoustic detectors, no transcript). The detectors miss some events, so these are lower bounds. A long gap between 'Hey' and 'Google' also counts, so it is not always a block."));
    chartPattern(card, pat); box.appendChild(card);
    card = el("div", "card"); card.appendChild(el("h2", "", "Inside each recording"));
    const sel = el("select"); PH_ORDER.forEach(p => { const o = el("option", "", PH_LABEL[p]); o.value = p; sel.appendChild(o); });
    card.appendChild(sel); const holder = el("div"); card.appendChild(holder);
    const draw = () => { holder.innerHTML = ""; chartSegments(holder, pat, sel.value); }; sel.addEventListener("change", draw); draw();
    card.appendChild(el("p", "hint", "Each row is one speaker's recording of that phrase. Teal bars are stretches of voice; red marks silent gaps of 0.25 s or more. Several short bars close together are repetitions or restarts."));
    box.appendChild(card);
  } else { card.appendChild(el("p", "empty", "Pattern analysis unavailable.")); box.appendChild(card); }

  if (ds) {
    card = el("div", "card"); card.appendChild(el("h2", "", "What the model was trained on"));
    card.appendChild(el("p", "hint", ds.total + " clips, " + ds.hours + " h, in the generated training dataset, plus the " + clips.length + " real recordings."));
    barsFromCounts(card, "clips by label", ds.by_label, cssVar("--primary")); barsFromCounts(card, "clips by speaker kind", ds.by_speaker_kind, cssVar("--fluent")); box.appendChild(card);
  }

  card = el("div", "card"); card.appendChild(el("h2", "", "Every recording"));
  const t = el("table"), th = el("thead"), hr = el("tr");
  ["Clip", "Speaker", "Score in-sample", "Score held-out", "Decision now", "Disfluency found", "Why not accepted"].forEach((x, i) => hr.appendChild(el("th", i > 1 && i < 4 ? "num" : "", x))); th.appendChild(hr); t.appendChild(th);
  const hmap = {}; hc.forEach(x => hmap[x.id] = x);
  const tb = el("tbody");
  clips.slice().sort((a, b) => PH_ORDER.indexOf(a.phrase) - PH_ORDER.indexOf(b.phrase) || a.speaker.localeCompare(b.speaker)).forEach(x => {
    const r = el("tr"), ok = x.score >= thr, h2 = hmap[x.id];
    r.append(el("td", "", x.phrase), el("td", "", x.speaker), el("td", "num", x.score.toFixed(2)), el("td", "num", h2 ? h2.score.toFixed(2) : "-"),
      el("td", "", x.wake ? (ok ? "accepted" : "missed") : (ok ? "false accept" : "rejected")), el("td", "", x.events.join(", ") || "-"), el("td", "", ok ? "" : (x.reason || "score below threshold")));
    tb.appendChild(r);
  }); t.appendChild(tb); const wr = el("div", "tableWrap"); wr.appendChild(t); card.appendChild(wr); box.appendChild(card);
  if (heldErr) { const n = el("p", "hint", "Held-out results unavailable: " + heldErr); box.appendChild(n); }
}
$("#checkRun").addEventListener("click", loadCheck);
