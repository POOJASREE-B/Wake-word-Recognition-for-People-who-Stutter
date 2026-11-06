/* Wake word recognition for speech disfluency: frontend (vanilla JS, no dependencies).
   Same backend endpoints as before: /api/health, /api/detect, /api/dataset/*, /api/synthesize, /api/model/*. */
"use strict";
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
const CLASSES = ["bg", "hey", "goo", "gle", "other"];
const SR = 16000;
const WINDOW_S = 6;
// pause after the phrase before an accept is final: PREFS.tail (Settings), default 0.3 s
const REJECT_TAIL_S = 1.2;      // longer pause before a rejection is final, so blocks and long pauses are never cut off
const RESET_SILENCE_S = 2.5;    // quiet after an undetected attempt before it is dropped (grid: 21/23 vs 20/23 found, same false alarms)
const MONO = '11px ui-monospace, Consolas, monospace';

/* ------------------------------------------------------------------ helpers */
function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; }
function tune(res) {                                  // Settings > Sensitivity moves the decision threshold of this page only
  if (res && res.span && !res.reason && res.score != null && res.threshold != null && PREFS.sens) {
    const thr = res.threshold + PREFS.sens;
    res.detected = res.score >= thr; res.soft = res.score >= thr - (res.soft_margin || 0.08); res.threshold = Math.round(thr * 1000) / 1000;
  }
  return res;
}
async function api(path, opts) {
  const r = await fetch(path, opts);
  let data = null;
  try { data = await r.json(); } catch (e) { /* not json */ }
  if (!r.ok) throw new Error((data && data.error) || ("HTTP " + r.status));
  return data;
}
function b64ToBlob(b64, type) {
  const bin = atob(b64); const arr = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
  return new Blob([arr], { type });
}
function encodeWav(f32, rate) {
  const buf = new ArrayBuffer(44 + f32.length * 2); const v = new DataView(buf);
  const w = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  w(0, "RIFF"); v.setUint32(4, 36 + f32.length * 2, true); w(8, "WAVE"); w(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, rate, true); v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  w(36, "data"); v.setUint32(40, f32.length * 2, true);
  for (let i = 0; i < f32.length; i++) { const s = Math.max(-1, Math.min(1, f32[i])); v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true); }
  return new Blob([buf], { type: "audio/wav" });
}
function downsample(buf, inRate, outRate) {
  if (inRate === outRate) return buf;
  const ratio = inRate / outRate, n = Math.floor(buf.length / ratio), out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const s = Math.floor(i * ratio), e = Math.min(buf.length, Math.max(s + 1, Math.floor((i + 1) * ratio)));
    let sum = 0; for (let j = s; j < e; j++) sum += buf[j];
    out[i] = sum / (e - s);
  }
  return out;
}
async function decodeBlob(blob) {
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  try {
    const ab = await blob.arrayBuffer();
    const audio = await ctx.decodeAudioData(ab);
    return { samples: audio.getChannelData(0).slice(), rate: audio.sampleRate };
  } finally { ctx.close(); }
}
function playBlob(blob) { const a = new Audio(URL.createObjectURL(blob)); a.play(); return a; }
function beep() {
  if (!PREFS.beep) return;
  try {
    const c = new (window.AudioContext || window.webkitAudioContext)(); const o = c.createOscillator(); const g = c.createGain();
    o.frequency.value = 880; g.gain.value = 0.06; o.connect(g); g.connect(c.destination); o.start();
    setTimeout(() => { o.stop(); c.close(); }, 140);
  } catch (e) { /* ignore */ }
}
function rmsOf(a) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * a[i]; return Math.sqrt(s / Math.max(1, a.length)); }
function concat(chunks) { let n = 0; chunks.forEach(c => n += c.length); const o = new Float32Array(n); let p = 0; chunks.forEach(c => { o.set(c, p); p += c.length; }); return o; }
function cssVar(n) { return getComputedStyle(document.documentElement).getPropertyValue(n).trim() || "#888"; }
function hexRgb(h) { h = h.replace("#", ""); if (h.length === 3) h = h.split("").map(c => c + c).join(""); return [0, 2, 4].map(i => parseInt(h.slice(i, i + 2), 16)); }
function palette() {
  return { bg: cssVar("--ink-muted"), hey: cssVar("--primary"), goo: cssVar("--fluent"), gle: cssVar("--listening"), other: cssVar("--disfluent") };
}
const pct = (v, d = 1) => (v === undefined || v === null || isNaN(v)) ? "-" : (v * 100).toFixed(d) + "%";
const num = (v, d = 2) => (v === undefined || v === null || isNaN(v)) ? "-" : Number(v).toFixed(d);
function eventTag(ev) {
  const e = el("span", "ev " + ev.type);
  let t = ev.type; if (ev.word) t += " " + ev.word;
  if (ev.count) t += " x" + ev.count; if (ev.duration_ms) t += " " + ev.duration_ms + " ms";
  e.textContent = t; return e;
}
function cmdBlock(cmd) {
  const wrap = el("div", "cmd"); const c = el("code", "", cmd);
  const b = el("button", "btn", "Copy"); b.type = "button"; b.dataset.copy = cmd;
  wrap.append(c, b); return wrap;
}
document.addEventListener("click", ev => {
  const b = ev.target.closest("[data-copy]"); if (!b) return;
  if (navigator.clipboard) navigator.clipboard.writeText(b.dataset.copy).then(() => { b.textContent = "Copied"; setTimeout(() => { b.textContent = "Copy"; }, 1500); }).catch(() => { });
});

/* ------------------------------------------------------------------ theme (the choice itself lives in prefs.js) */
const LAST = { timeline: null };
function redrawForTheme() {                           // called by prefs.js when the theme changes: canvases read their colours from CSS variables
  try {
    if (LAST.timeline) drawTimeline($("#liveCanvas"), ...LAST.timeline);
    if (typeof DS !== "undefined" && DS.cur) { drawAlignedFor(DS.cur); if (!$("#specCanvas").hidden) drawSpec($("#specCanvas"), DS.cur.d); }
    const lg = $("#legend"); if (lg) $$("i", lg).forEach(i => { i.style.background = palette()[i.dataset.k]; });
  } catch (e) { /* drawing is optional */ }
}

/* ------------------------------------------------------------------ tabs + health */
$$(".navbtn[data-tab]").forEach(b => b.addEventListener("click", () => {
  $$(".navbtn[data-tab]").forEach(x => x.classList.toggle("active", x === b));
  $$(".panel").forEach(p => p.classList.toggle("active", p.id === "tab-" + b.dataset.tab));
  if (b.dataset.tab === "data") initDataset();
  if (b.dataset.tab === "report") loadReport();
  if (b.dataset.tab === "check" && typeof loadCheck === "function") loadCheck();
}));
const APP = { threshold: null };
(async function health() {
  const box = $("#conn"), txt = $("#connText");
  try {
    const h = await api("/api/health");
    if (!h.model_loaded) { txt.textContent = "Service unavailable"; box.className = "conn warn"; box.title = "The detection model is not loaded on this server. Please try again later."; }
    else {
      txt.textContent = "Ready"; box.className = "conn ok";
      box.title = "Model " + (h.model_name || "") + ", threshold " + num(h.threshold);
      APP.threshold = h.threshold; $("#thrText").textContent = num(h.threshold);
      if (h.research) $$(".navbtn.research").forEach(b => { b.hidden = false; });
    }
  } catch (e) { txt.textContent = "Offline"; box.className = "conn bad"; box.title = "The server cannot be reached. Please try again later."; }
})();
$("#legend").innerHTML = "";
[["hey", "Hey"], ["goo", "Goo"], ["gle", "gle"], ["other", "Other speech"], ["bg", "Silence or pause"]].forEach(([k, t]) => {
  const s = el("span"); const i = el("i"); i.style.background = palette()[k]; i.dataset.k = k; s.append(i, t); $("#legend").appendChild(s);
});

/* ------------------------------------------------------------------ mic state */
const MIC = { state: "idle", active: false, decidedAt: 0, lastLoud: 0, levels: [] };
const MIC_LABEL = {
  idle: "Press Start, then say Hey Google.",
  listening: "Listening. Take your time.",
  hearing: "Hearing you",
  accepted: "Wake word detected",
  blocked: "Microphone access is blocked"
};
function stage(t) { const e = $("#dStatus"); if (e) e.textContent = t; }
function setMic(state, detail, labelOverride) {
  MIC.state = state;
  $("#micWell").dataset.state = state;
  const label = labelOverride || MIC_LABEL[state] || "";
  $("#micLabel").textContent = label;
  const d = $("#micDetail"); d.textContent = detail || ""; d.classList.toggle("msg-error", state === "blocked");
  stage(state === "listening" ? "Waiting for speech" : label);
}
function micError(e) {
  if (e && (e.name === "NotAllowedError" || e.name === "SecurityError" || e.name === "PermissionDeniedError")) {
    setMic("blocked", "Click the lock icon in the address bar, set Microphone to Allow, then reload this page. The microphone works only on a secure (https) address.");
  } else if (e && e.name === "NotFoundError") {
    setMic("blocked", "No microphone was found. Connect one and reload this page.", "No microphone found");
  } else {
    setMic("blocked", (e && e.message) || "The microphone could not be opened.", "Microphone could not start");
  }
}
function drawStrip() {
  const cv = $("#stripCanvas"), ctx = cv.getContext("2d"), W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  const bw = 4, n = Math.floor(W / bw), lv = MIC.levels.slice(-n);
  ctx.fillStyle = MIC.state === "hearing" ? cssVar("--listening") : cssVar("--ink-muted");
  lv.forEach((v, i) => { const h = Math.max(2, Math.min(1, v) * (H - 8)); ctx.fillRect((n - lv.length + i) * bw, (H - h) / 2, bw - 1, h); });
}
function onLevel(c) {
  const r = rmsOf(c), now = performance.now();
  MIC.levels.push(Math.min(1, r * 6)); if (MIC.levels.length > 400) MIC.levels.shift();
  $("#micWell").style.setProperty("--lvl", Math.min(1, r * 8).toFixed(2));
  const loud = r > 0.015; if (loud) MIC.lastLoud = now;
  if (MIC.active) {
    if (MIC.state === "listening" && loud) setMic("hearing");
    else if (MIC.state === "hearing" && !loud && now - MIC.lastLoud > 700) setMic("listening");
    else if ((MIC.state === "accepted" || MIC.state === "rejected") && loud && now - MIC.decidedAt > 1500) setMic("hearing");
  }
  drawStrip();
}
function clearStrip() { MIC.levels = []; drawStrip(); $("#micWell").style.setProperty("--lvl", "0"); }

/* ------------------------------------------------------------------ recent attempts */
const ATT = [];
function typeOf(res) { const t = Array.from(new Set((res.events || []).map(e => e.type))); return t.length ? t.join(", ") : "none"; }
function addAttempt(res) {
  ATT.unshift({ t: new Date().toLocaleTimeString(), ok: !!res.detected, c: res.score, type: typeOf(res) });
  if (ATT.length > 30) ATT.pop();
  const tb = $("#attTable tbody"); tb.innerHTML = "";
  ATT.forEach(a => {
    const tr = el("tr");
    const r = el("td"); r.appendChild(el("span", "tag " + (a.ok ? "acc" : "rej"), a.ok ? "Accepted" : "Not the wake word"));
    const t0 = el("td", "mono", a.t), c = el("td", "mono", num(a.c)), ty = el("td", "", a.type);
    tr.append(t0, r, c, ty); tb.appendChild(tr);
  });
  $("#attEmpty").hidden = true;
}

/* ------------------------------------------------------------------ timeline drawing */
function drawTimeline(canvas, samples, rate, res) {
  LAST.timeline = [samples, rate, res];
  const ctx = canvas.getContext("2d"), W = canvas.width, H = canvas.height, C = palette();
  ctx.clearRect(0, 0, W, H);
  const dur = samples ? samples.length / rate : (res && res.duration) || 1;
  const X = t => (t / dur) * W;
  const mut = cssVar("--ink-muted"), line = cssVar("--line"), surf = cssVar("--surface");
  const wfH = 100, curveTop = 112, curveH = 90, bandTop = 214, bandH = 26;
  const UNIT = { H: C.hey, G: C.goo, L: C.gle };
  ctx.font = MONO;
  if (samples) {
    const step = Math.max(1, Math.floor(samples.length / W));
    ctx.fillStyle = mut;
    for (let x = 0; x < W; x++) {
      let mn = 1, mx = -1;
      for (let j = x * step; j < Math.min(samples.length, (x + 1) * step); j++) { const v = samples[j]; if (v < mn) mn = v; if (v > mx) mx = v; }
      if (mx < mn) continue;
      const peak = 0.9; ctx.fillRect(x, wfH / 2 - Math.min(mx / peak, 1) * wfH / 2, 1, Math.max(1, (mx - mn) / peak * wfH / 2));
    }
  }
  if (res && res.curves) {
    const p = res.curves.p, st = res.curves.step_s;
    ctx.fillStyle = line; ctx.fillRect(0, curveTop + curveH, W, 1);
    [["hey", 1], ["goo", 2], ["gle", 3], ["other", 4]].forEach(([name, idx]) => {
      ctx.strokeStyle = C[name]; ctx.lineWidth = 1.6; ctx.beginPath();
      p.forEach((row, i) => { const x = X(i * st), y = curveTop + curveH - row[idx] * curveH; i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
      ctx.stroke();
    });
  }
  if (res && res.units) {
    res.units.forEach(u => {
      ctx.fillStyle = UNIT[u.kind] || line;
      ctx.fillRect(X(u.start), bandTop, Math.max(2, X(u.end) - X(u.start)), bandH);
      if (X(u.end) - X(u.start) > 26 && u.label !== "pause") { ctx.fillStyle = surf; ctx.fillText(u.label, X(u.start) + 4, bandTop + 17); }
    });
  }
  if (res && res.span) {
    ctx.strokeStyle = res.detected ? cssVar("--accepted") : cssVar("--rejected"); ctx.lineWidth = 2; ctx.setLineDash([5, 3]);
    ctx.strokeRect(X(res.span.start), bandTop - 4, Math.max(2, X(res.span.end) - X(res.span.start)), bandH + 8); ctx.setLineDash([]);
  }
  ctx.fillStyle = mut;
  for (let s = 0; s <= dur; s += dur > 8 ? 2 : 1) { ctx.fillRect(X(s), H - 18, 1, 5); ctx.fillText(s + "s", X(s) + 3, H - 6); }
  ctx.fillText("waveform", 4, 12); ctx.fillText("frame model probabilities", 4, curveTop + 10); ctx.fillText("decoded Hey, Goo, gle path", 4, bandTop - 6);
  $$("#legend i").forEach(i => { i.style.background = C[i.dataset.k]; });
}

/* ------------------------------------------------------------------ detection card + decision */
const live = { origBlob: null, fluentBlob: null };
/* "Sounds heard" is refreshed every 0.7 s, too fast to read. A heard text stays on screen for HEARD_HOLD_MS, and every
   new, different text is also kept in a log (newest first) so earlier attempts can be read back. */
const HEARD_HOLD_MS = 4000, HEARD_LOG_MAX = 60;
const heardState = { text: "", shownAt: 0, last: "" };
function showHeard(text, reason, res) {
  const now = performance.now(), el = $("#heardText");
  if (text) {
    el.textContent = text; heardState.text = text; heardState.shownAt = now;
    if (text !== heardState.last) { heardState.last = text; addHeardLog(text, res); }
  } else if (!heardState.text || now - heardState.shownAt > HEARD_HOLD_MS) {
    el.textContent = reason || "No wake-word sounds yet"; heardState.text = "";
  }
}
function addHeardLog(text, res) {
  const ol = $("#heardLog"); if (!ol) return;
  const li = document.createElement("li");
  const t = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  const tag = res && res.detected ? "phrase found, not yet confirmed" : "not the wake word";
  li.innerHTML = '<time></time><span class="h"></span><em></em>';
  li.querySelector("time").textContent = t;
  li.querySelector(".h").textContent = text;
  li.querySelector("em").textContent = tag + (res && res.score != null ? " · " + num(res.score) : "");
  li.className = "";
  ol.prepend(li);
  while (ol.children.length > HEARD_LOG_MAX) ol.lastChild.remove();
  const e = $("#heardEmpty"); if (e) e.hidden = true;
}
function clearHeardLog() {
  $("#heardLog").innerHTML = ""; heardState.last = ""; heardState.text = "";
  const e = $("#heardEmpty"); if (e) e.hidden = false;
}
document.addEventListener("click", ev => { if (ev.target && ev.target.id === "btnClearHeard") clearHeardLog(); });
function readout(res) {
  const sounds = (res.units || []).filter(u => ["H", "G", "L"].includes(u.kind)).map(u => u.label).join(" · ");
  showHeard(res.heard_text || (res.span ? sounds : ""), res.reason, res);
  $("#scoreText").textContent = num(res.score);
  $("#thrText").textContent = num(res.threshold);
  $("#tailText").textContent = res.tail_silence_s == null ? "-" : res.tail_silence_s.toFixed(1) + " s (needs " + PREFS.tail.toFixed(1) + " s)";
  $("#gaugeFill").style.width = Math.round((res.score || 0) * 100) + "%";
  $("#gaugeFill").style.background = res.detected ? cssVar("--accepted") : cssVar("--ink-muted");
  $("#gaugeThr").style.left = Math.round((res.threshold || 0) * 100) + "%";
  const box = $("#eventsBox"); box.innerHTML = "";
  (res.events || []).forEach(e => box.appendChild(eventTag(e)));
  if (!(res.events || []).length) box.textContent = res.span ? (res.detected ? "none (fluent)" : "none") : "-";
}
async function showResult(res, blob, samples, rate) {
  readout(res);
  $("#fluentText").textContent = res.detected ? (res.fluent_output || "Hey Google") : "-";
  live.origBlob = blob; live.fluentBlob = res.repaired_wav_b64 ? b64ToBlob(res.repaired_wav_b64, "audio/wav") : null;
  $("#playOrig").disabled = !blob; $("#playFluent").disabled = !res.detected;
  if (!samples && blob) { try { const d = await decodeBlob(blob); samples = d.samples; rate = d.rate; } catch (e) { /* ignore */ } }
  drawTimeline($("#liveCanvas"), samples, rate || SR, res);
}
/* Spoken confirmation: the system says "Wake word detected" only when the wake word was accepted. */
let _utter = null;                                  // keep a reference: Chrome drops an utterance that is garbage-collected before it ends
function announce(text) {
  if (!PREFS.voice) return;
  try { L.speakUntil = performance.now() + 2200; } catch (e) { /* ignore */ }
  try {
    if (!window.speechSynthesis) return;
    speechSynthesis.cancel(); speechSynthesis.resume();
    _utter = new SpeechSynthesisUtterance(text); _utter.lang = "en-US"; _utter.rate = 1.0; _utter.volume = 1;
    const v = speechSynthesis.getVoices().find(x => /^en/i.test(x.lang)); if (v) _utter.voice = v;
    speechSynthesis.speak(_utter);
  } catch (e) { /* the beep and the on-screen label still tell the person */ }
}
if (window.speechSynthesis) { speechSynthesis.getVoices(); speechSynthesis.addEventListener && speechSynthesis.addEventListener("voiceschanged", () => speechSynthesis.getVoices()); }
function decide(res, stillListening) {
  MIC.decidedAt = performance.now();
  if (res.detected) {
    setMic("accepted", "Matched with confidence " + num(res.score), "Wake word detected");
    beep(); announce("Wake word detected"); addAttempt(res);
    if (typeof addHeardLog === "function") { heardState.last = ""; const ol = $("#heardLog"); if (ol) { const li = document.createElement("li"); li.className = "ok"; li.innerHTML = '<time></time><span class="h"></span><em></em>'; li.querySelector("time").textContent = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }); li.querySelector(".h").textContent = "WAKE WORD DETECTED"; li.querySelector("em").textContent = "score " + num(res.score); ol.prepend(li); const e = $("#heardEmpty"); if (e) e.hidden = true; } }
    if (stillListening) setTimeout(() => { if (MIC.active && MIC.state === "accepted") setMic("listening"); }, 2500);
  } else if (stillListening) {
    /* not the wake word: stay idle and silent, keep listening */
  } else {
    setMic("idle", "", "No wake word detected");      // an explicitly analysed clip still gets a quiet answer
  }
}
$("#playOrig").addEventListener("click", () => live.origBlob && playBlob(live.origBlob));
$("#playFluent").addEventListener("click", () => {
  if (live.fluentBlob) playBlob(live.fluentBlob);
  else if (window.speechSynthesis) { const u = new SpeechSynthesisUtterance("Hey Google"); u.lang = "en-US"; speechSynthesis.speak(u); }
});

/* ------------------------------------------------------------------ microphone */
class Mic {
  async start(onChunk) {
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
    this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    this.src = this.ctx.createMediaStreamSource(this.stream);
    this.proc = this.ctx.createScriptProcessor(4096, 1, 1);
    this.proc.onaudioprocess = e => onChunk(downsample(new Float32Array(e.inputBuffer.getChannelData(0)), this.ctx.sampleRate, SR));
    this.src.connect(this.proc); this.proc.connect(this.ctx.destination);
  }
  stop() {
    try { this.proc.disconnect(); this.src.disconnect(); } catch (e) { /* ignore */ }
    if (this.stream) this.stream.getTracks().forEach(t => t.stop());
    if (this.ctx) this.ctx.close();
  }
}
async function capture(onLevel) {
  const mic = new Mic(), chunks = [];
  await mic.start(c => { chunks.push(c); if (onLevel) onLevel(c); });
  return { stop() { mic.stop(); return concat(chunks); } };
}

/* ---- continuous listening: no time limit, the person decides when to stop */
const L = { mic: null, chunks: [], len: 0, total: 0, timer: null, inflight: false, lastEnd: -10, pending: null };
function ringPush(c) {
  L.chunks.push(c); L.len += c.length; L.total += c.length;
  while (L.len - L.chunks[0].length > WINDOW_S * SR) { L.len -= L.chunks[0].length; L.chunks.shift(); }
  onLevel(c);
}
function whyNot(res) {
  if (res.score != null && res.threshold != null && res.score < res.threshold)
    return "score " + num(res.score) + " is below the threshold " + num(res.threshold) + (res.reason ? " (" + res.reason + ")" : "");
  return res.reason || "the phrase did not match";
}
async function listenTick() {
  if (L.inflight || !L.mic) return;
  // while the system is speaking its own confirmation, do not analyse (the mic would hear the loudspeaker)
  if (performance.now() < (L.speakUntil || 0)) { L.chunks = []; L.len = 0; return; }   // own timer: speechSynthesis.speaking can stay true in Chrome
  const buf = concat(L.chunks);
  if (buf.length < SR * 0.8) return;
  const recent = buf.slice(Math.max(0, buf.length - SR * 4));
  let peak = 0; for (let i = 0; i + 320 <= recent.length; i += 320) peak = Math.max(peak, rmsOf(recent.subarray(i, i + 320)));
  if (peak < 0.004) { if (MIC.state === "listening") stage("Waiting for speech"); return; }
  // Each attempt is judged on its own: once an attempt is followed by RESET_SILENCE_S of silence without being
  // accepted, drop it, so a failed 'Hey Google' is never stitched to the next attempt.
  const tailN = Math.floor(SR * RESET_SILENCE_S);
  if (buf.length > tailN + SR * 0.3) {
    const tail = buf.subarray(buf.length - tailN);
    let tailPeak = 0; for (let i = 0; i + 320 <= tail.length; i += 320) tailPeak = Math.max(tailPeak, rmsOf(tail.subarray(i, i + 320)));
    // a candidate that is waiting for its confirming check is kept (v1.4): a slow check must not erase it
    if (tailPeak < Math.max(0.004, 0.05 * peak) && L.pending === null) { L.chunks = []; L.len = 0; return; }
  }
  L.inflight = true;
  stage("Checking the last " + Math.min(6, buf.length / SR).toFixed(1) + " s of audio");
  const startTime = (L.total - buf.length) / SR; const blob = encodeWav(buf, SR);
  try {
    const res = tune(await api("/api/detect?repair=1", { method: "POST", headers: { "Content-Type": "audio/wav" }, body: blob }));
    if (!L.mic) return;
    readout(res);
    stage(!res.span ? "No wake-word pattern in this audio" : res.detected ? "Phrase found, confirming" : "Speech heard, not the wake word yet");
    const absEnd = res.span ? startTime + res.span.end : -1;
    const tail = res.tail_silence_s;
    const fresh = absEnd > L.lastEnd + 0.5;
    // v1.3: a wake-up is accepted only when the SAME phrase is found on two consecutive checks (0.7 s apart);
    // in replay tests on 12 min of continuous talk this halves the false wake-ups and costs one tick of latency
    // v1.5: the FIRST check must reach the threshold; the CONFIRMING check (same phrase, 0.7 s later) may dip a little below it
    // (res.soft: within SOFT_MARGIN of the threshold, all other gates passed). Scores of real stuttered speech sit close to the
    // threshold, so requiring two clean crossings in a row dropped many true wake-ups.
    // v1.6: a candidate is remembered as soon as the phrase scores above the threshold, even while the person is still
    // finishing (before: it only counted once a 0.3 s pause had already followed, so a score that crossed the threshold
    // while the last sound was fading was thrown away and the next check often fell below it). The wake-up is accepted
    // when the same phrase is found on a second check (within 0.8 s of the first) AND the pause has been reached.
    const nowMs = performance.now();
    if (L.cand && nowMs - L.cand.t > 4000) L.cand = null;
    const hit = !!res.span && fresh && (res.detected || res.soft);
    if (hit && L.cand && Math.abs(absEnd - L.cand.end) < 0.8) { L.cand.end = absEnd; L.cand.n++; L.cand.t = nowMs; }
    else if (res.detected && res.span && fresh) L.cand = { end: absEnd, n: 1, t: nowMs };
    const paused = tail != null && tail >= PREFS.tail;
    const acceptFinal = !!L.cand && L.cand.n >= 2 && paused && hit;
    L.pending = (L.cand && !acceptFinal) ? L.cand.end : null;
    if (acceptFinal) L.cand = null;
    const rejectFinal = !res.detected && res.span && tail != null && tail >= REJECT_TAIL_S && fresh;
    if (acceptFinal) {
      L.lastEnd = absEnd; L.chunks = []; L.len = 0;   // do not reuse the same audio for the next decision
      res.detected = true;                            // accepted on the confirming check (it may have scored just under the threshold)
      decide(res, true);                              // announce FIRST: drawing must never be able to swallow the confirmation
      try { await showResult(res, blob, buf, SR); } catch (e) { /* drawing problems must not matter */ }
    } else if (rejectFinal) {
      L.lastEnd = absEnd;                               // not the wake word: nothing is SAID, but the reason is shown
      showResult(res, blob, buf, SR);
      if (MIC.state !== "accepted") $("#micDetail").textContent = "Last attempt not accepted: " + whyNot(res);
    } else if (res.detected) {
      showResult(res, blob, buf, SR);                  // not awaited: drawing must never delay the confirming check
      if (MIC.state !== "accepted") $("#micDetail").textContent = (tail != null && tail < PREFS.tail)
        ? "Phrase found (score " + num(res.score) + " vs threshold " + num(res.threshold) + "). Pause for a moment so I know you are done."
        : "Phrase found (score " + num(res.score) + " vs threshold " + num(res.threshold) + "). Confirming…";
    } else if (res.span && MIC.state !== "accepted") {
      $("#micDetail").textContent = "Hearing speech, not the wake word yet (score " + num(res.score) + " vs threshold " + num(res.threshold) + ")";
    }
  } catch (e) { stopListening(); $("#micDetail").textContent = "Error: " + e.message; $("#micDetail").classList.add("msg-error"); }
  finally { L.inflight = false; }
}
async function startListening() {
  try {
    clearStrip(); L.chunks = []; L.len = 0; L.total = 0; L.lastEnd = -10; L.pending = null; L.cand = null;
    L.mic = new Mic(); await L.mic.start(ringPush);
    MIC.active = true; L.timer = setInterval(listenTick, 700);
    $("#btnStart").disabled = true; $("#btnStop").disabled = false; $("#btnRecord").disabled = true;
    setMic("listening");
  } catch (e) { L.mic = null; MIC.active = false; micError(e); }
}
function stopListening() {
  clearInterval(L.timer); if (L.mic) L.mic.stop(); L.mic = null; MIC.active = false;
  $("#btnStart").disabled = false; $("#btnStop").disabled = true; $("#btnRecord").disabled = false;
  clearStrip(); setMic("idle");
}
$("#btnStart").addEventListener("click", startListening);
$("#btnStop").addEventListener("click", stopListening);

/* ---- record a clip: runs until the person presses stop */
const REC = { cap: null };
async function analyse(blob, buf, name) {
  const res = tune(await api("/api/detect?repair=1", { method: "POST", headers: { "Content-Type": blob.type || "application/octet-stream" }, body: blob }));
  await showResult(res, blob, buf || null, buf ? SR : null); decide(res, false);
}
$("#btnRecord").addEventListener("click", async () => {
  const b = $("#btnRecord");
  if (REC.cap) {
    const cap = REC.cap; REC.cap = null; MIC.active = false;
    const buf = cap.stop(); b.textContent = "Record a clip"; b.classList.remove("rec");
    $("#btnStart").disabled = false; clearStrip();
    try { setMic("idle", "Analysing the recording"); await analyse(encodeWav(buf, SR), buf); } catch (e) { setMic("idle", "Error: " + e.message); }
    return;
  }
  try {
    clearStrip(); REC.cap = await capture(onLevel); MIC.active = true;
    b.textContent = "Stop and analyse"; b.classList.add("rec"); $("#btnStart").disabled = true;
    setMic("listening");
  } catch (e) { REC.cap = null; MIC.active = false; micError(e); }
});
$("#fileIn").addEventListener("change", async ev => {
  const f = ev.target.files[0]; if (!f) return;
  if (L.mic) stopListening();
  try { setMic("idle", "Analysing " + f.name); await analyse(f, null, f.name); } catch (e) { setMic("idle", "Error: " + e.message); }
  ev.target.value = "";
});

/* ------------------------------------------------------------------ dataset */
const DS = { page: 1, per: 40, total: 0, inited: false, cur: null, items: [], idx: -1, audio: null };
function cmdEmpty(parent, text) { parent.innerHTML = ""; parent.appendChild(el("p", "empty", text)); }
async function initDataset() {
  if (DS.inited) return; DS.inited = true;
  const tb = $("#dsTable tbody");
  for (let i = 0; i < 6; i++) { const tr = el("tr"); const td = el("td"); td.colSpan = 5; td.appendChild(el("div", "skel w80")); tr.appendChild(td); tb.appendChild(tr); }
  try {
    const s = await api("/api/dataset/summary");
    $("#dsCount").textContent = "Clips: " + s.total.toLocaleString() + "  |  Hours: " + s.hours;
    const box = $("#dsSummary"); box.innerHTML = "";
    Object.entries(s.by_label).forEach(([k, v]) => box.appendChild(el("span", "tag " + (k === "negative" ? "negative" : k), k + ": " + v)));
    Object.entries(s.by_split).forEach(([k, v]) => box.appendChild(el("span", "tag", k + ": " + v)));
    if (!s.total) { DS.inited = false; cmdEmpty($("#dsTable").parentElement.parentElement, "The dataset is not available on this server."); return; }
  } catch (e) {
    DS.inited = false;
    cmdEmpty($("#dsTable").parentElement.parentElement, "The dataset is not available on this server.");
    return;
  }
  await loadList(1);
}
async function loadList(page) {
  const q = new URLSearchParams({ page, per_page: DS.per, split: $("#fSplit").value, label: $("#fLabel").value, category: $("#fCat").value, q: $("#fQ").value });
  try {
    const d = await api("/api/dataset/samples?" + q.toString());
    DS.page = d.page; DS.total = d.total; DS.items = d.items; DS.idx = -1;
    const sel = $("#fCat"); if (sel.options.length <= 1) d.categories.forEach(c => { const o = el("option", "", c); o.value = c; sel.appendChild(o); });
    const tb = $("#dsTable tbody"); tb.innerHTML = "";
    if (!d.items.length) { const tr = el("tr"); const td = el("td", "empty", "No clips match these filters."); td.colSpan = 5; tr.appendChild(td); tb.appendChild(tr); }
    d.items.forEach((it, i) => {
      const tr = el("tr"); tr.tabIndex = 0; tr.dataset.i = i;
      const neg = it.label === "negative";
      const cls = el("td"); cls.appendChild(el("span", "tag " + (neg ? "negative" : "acc"), neg ? "Negative" : "Wake word"));
      const kind = el("td"); kind.appendChild(el("span", "tag " + it.label, it.label));
      tr.append(el("td", "", it.sample_id), cls, kind, el("td", "", it.category), el("td", "num", String(it.duration_seconds)));
      tr.addEventListener("click", () => selectRow(i));
      tr.addEventListener("keydown", e => { if (e.key === "Enter") selectRow(i); });
      tb.appendChild(tr);
    });
    $("#pgInfo").textContent = "Page " + d.page + " of " + Math.max(1, Math.ceil(d.total / d.per_page)) + "  (" + d.total.toLocaleString() + " clips)";
  } catch (e) { $("#pgInfo").textContent = e.message; }
}
function selectRow(i) {
  if (i < 0 || i >= DS.items.length) return;
  DS.idx = i; $$("#dsTable tbody tr").forEach(r => r.classList.toggle("sel", Number(r.dataset.i) === i));
  const r = $$("#dsTable tbody tr")[i]; if (r) r.scrollIntoView({ block: "nearest" });
  showSample(DS.items[i].sample_id);
}
$("#fGo").addEventListener("click", () => loadList(1));
$("#fQ").addEventListener("keydown", e => { if (e.key === "Enter") loadList(1); });
$("#pgPrev").addEventListener("click", () => DS.page > 1 && loadList(DS.page - 1));
$("#pgNext").addEventListener("click", () => DS.page * DS.per < DS.total && loadList(DS.page + 1));

function eventSpan(e) {
  const s = Number(e.start) || 0;
  const end = e.end !== undefined ? Number(e.end) : (e.duration_ms ? s + e.duration_ms / 1000 : s + 0.08);
  return [s, Math.max(end, s + 0.04)];
}
function drawLane(ctx, d, top, h, W, shared, color, label, shade) {
  const C = palette(), X = t => (t / shared) * W, mid = top + h / 2 - 8, amp = (h - 28) / 2;
  ctx.fillStyle = cssVar("--ink-muted"); ctx.font = MONO; ctx.fillText(label, 6, top + 12);
  ctx.fillStyle = color;
  const w = Math.max(1, X(d.duration) / d.peaks.length);
  d.peaks.forEach((p, i) => { const x = (i / d.peaks.length) * X(d.duration); ctx.fillRect(x, mid - p[1] * amp, w, Math.max(1, (p[1] - p[0]) * amp)); });
  d.segments.forEach(([s, e, c]) => { ctx.fillStyle = C[CLASSES[c]] || cssVar("--line"); ctx.fillRect(X(s), top + h - 14, Math.max(1, X(e) - X(s)), 6); });
  if (shade) d.events.forEach(e => {
    const [s, en] = eventSpan(e), t = cssVar("--disfluent");
    ctx.fillStyle = t; ctx.globalAlpha = 0.22; ctx.fillRect(X(s), top + 16, X(en) - X(s), h - 32); ctx.globalAlpha = 1;
    ctx.fillStyle = t; ctx.fillRect(X(s), top + 16, 1, h - 32);
  });
}
function drawAligned(canvas, lanes) {
  const W = 640, laneH = 120, axis = 26;
  canvas.height = lanes.length * laneH + axis; canvas.width = W;
  const ctx = canvas.getContext("2d"); ctx.clearRect(0, 0, W, canvas.height);
  const shared = Math.max(...lanes.map(l => l.d.duration), 0.1);
  lanes.forEach((l, i) => {
    if (i) { ctx.fillStyle = cssVar("--line"); ctx.fillRect(0, i * laneH, W, 1); }
    drawLane(ctx, l.d, i * laneH, laneH, W, shared, l.color, l.label, l.shade);
  });
  ctx.fillStyle = cssVar("--ink-muted"); ctx.font = MONO;
  const stepS = shared > 4 ? 1 : 0.5, y = lanes.length * laneH;
  for (let s = 0; s <= shared; s += stepS) { const x = (s / shared) * W; ctx.fillRect(x, y + 4, 1, 5); ctx.fillText(s.toFixed(stepS < 1 ? 1 : 0) + "s", Math.min(x + 3, W - 28), y + 20); }
}
function drawAlignedFor(cur) {
  const lanes = [];
  if (cur.p) lanes.push({ d: cur.p, color: cssVar("--fluent"), label: "Fluent source " + cur.d.meta.fluent_pair_id, shade: false });
  const isNeg = cur.d.meta.label === "negative", isFl = cur.d.meta.label === "fluent";
  lanes.push({ d: cur.d, color: isFl ? cssVar("--fluent") : isNeg ? cssVar("--ink-muted") : cssVar("--disfluent"), label: isFl ? "Fluent clip" : isNeg ? "Negative clip" : "Disfluent version", shade: !isFl && !isNeg });
  drawAligned($("#alignCanvas"), lanes);
}
function drawSpec(canvas, d) {
  const ctx = canvas.getContext("2d"), W = canvas.width, H = canvas.height; ctx.clearRect(0, 0, W, H);
  const bytes = Uint8Array.from(atob(d.spec.b64), c => c.charCodeAt(0)), T = d.spec.frames, B = d.spec.bins;
  const lo = hexRgb(cssVar("--surface")), hi = hexRgb(cssVar("--primary"));
  const off = document.createElement("canvas"); off.width = T; off.height = B; const octx = off.getContext("2d"), img = octx.createImageData(T, B);
  for (let t = 0; t < T; t++) for (let b = 0; b < B; b++) {
    const v = Math.pow(bytes[t * B + b] / 255, 0.8), o = ((B - 1 - b) * T + t) * 4;
    for (let k = 0; k < 3; k++) img.data[o + k] = Math.round(lo[k] + (hi[k] - lo[k]) * v);
    img.data[o + 3] = 255;
  }
  octx.putImageData(img, 0, 0); ctx.imageSmoothingEnabled = true; ctx.drawImage(off, 0, 0, W, H);
}
function playClip(url) {
  if (DS.audio) { DS.audio.pause(); }
  const a = new Audio(url); a.playbackRate = $("#dSlow").checked ? 0.5 : 1; DS.audio = a; a.play();
}
function toggleClip() {
  if (DS.audio && !DS.audio.ended && DS.audio.currentTime > 0) { DS.audio.paused ? DS.audio.play() : DS.audio.pause(); }
  else if (DS.cur) playClip("/api/dataset/audio/" + DS.cur.id);
}
async function showSample(id) {
  try {
    const d = await api("/api/dataset/sample/" + id);
    let p = null; if (d.fluent_pair) { try { p = await api("/api/dataset/sample/" + d.fluent_pair); } catch (e) { /* pair missing */ } }
    d.meta.fluent_pair_id = d.fluent_pair || "";
    DS.cur = { id, d, p }; if (DS.audio) { DS.audio.pause(); DS.audio = null; }
    $("#detailEmpty").hidden = true; $("#detailBody").hidden = false; $("#detailTitle").textContent = id;
    drawAlignedFor(DS.cur);
    $("#specCanvas").hidden = true; $("#specBtn").textContent = "Show spectrogram";
    const m = d.meta, neg = m.label === "negative";
    const rows = [["Class", neg ? "Negative (should be rejected)" : "Wake word (should be accepted)"], ["Label", m.label], ["Category", m.category], ["Speaker", m.speaker_kind],
      ["Length", num(d.duration) + " s"], ["Source", m.source_id], ["Split", m.split], ["Context", m.context],
      ["Origin", m.provenance], ["Stuttering profile", m.stutter_profile], ["Emotion", m.emotion], ["Age group", m.age_group],
      ["Secondary behaviour", m.secondary_behaviour]];
    const dl = $("#detailMeta"); dl.innerHTML = "";
    rows.forEach(([k, v]) => { const r = el("div"); r.append(el("dt", "", k), el("dd", "mono", String(v === undefined ? "-" : v))); dl.appendChild(r); });
    const eb = $("#detailEvents"); eb.innerHTML = "";
    d.events.forEach(e => eb.appendChild(eventTag(e)));
    if (!d.events.length) eb.appendChild(el("span", "tag", m.is_wake_word === "1" ? "fluent, no disfluency map" : "not the wake word"));
    $("#dPlay").textContent = m.label === "disfluent" ? "Play disfluent" : "Play this clip";
    $("#dPlay").onclick = () => playClip("/api/dataset/audio/" + id);
    $("#dPlayPair").disabled = !d.fluent_pair; $("#dPlayPair").onclick = () => d.fluent_pair && playClip("/api/dataset/audio/" + d.fluent_pair);
  } catch (e) { $("#detailEmpty").hidden = false; $("#detailBody").hidden = true; $("#detailEmpty").textContent = "Could not load this clip: " + e.message; }
}
$("#specBtn").addEventListener("click", () => {
  const c = $("#specCanvas"); if (!DS.cur) return;
  c.hidden = !c.hidden; $("#specBtn").textContent = c.hidden ? "Show spectrogram" : "Hide spectrogram";
  if (!c.hidden) drawSpec(c, DS.cur.d);
});
document.addEventListener("keydown", e => {
  if (!$("#tab-data").classList.contains("active")) return;
  const t = e.target.tagName; if (t === "INPUT" || t === "SELECT" || t === "TEXTAREA" || e.ctrlKey || e.metaKey) return;
  if (e.key === "ArrowDown") { e.preventDefault(); selectRow(Math.min(DS.items.length - 1, DS.idx + 1)); }
  else if (e.key === "ArrowUp") { e.preventDefault(); selectRow(Math.max(0, DS.idx - 1)); }
  else if (e.key === " " && t !== "BUTTON") { e.preventDefault(); toggleClip(); }
  else if (e.key.toLowerCase() === "f" && DS.cur && DS.cur.d.fluent_pair) playClip("/api/dataset/audio/" + DS.cur.d.fluent_pair);
  else if (e.key.toLowerCase() === "s") { $("#dSlow").checked = !$("#dSlow").checked; if (DS.audio) DS.audio.playbackRate = $("#dSlow").checked ? 0.5 : 1; }
});
$("#dSlow").checked = !!PREFS.slow;
$("#dSlow").addEventListener("change", () => { if (DS.audio) DS.audio.playbackRate = $("#dSlow").checked ? 0.5 : 1; });

/* ------------------------------------------------------------------ voice lab */
const LAB = { blob: null, cap: null };
function labReady(blob, name) { LAB.blob = blob; $("#labGo").disabled = false; $("#labMsg").textContent = "Ready: " + name; }
$("#labRec").addEventListener("click", async () => {
  const b = $("#labRec");
  if (LAB.cap) { const buf = LAB.cap.stop(); LAB.cap = null; b.textContent = "Record"; b.classList.remove("rec"); labReady(encodeWav(buf, SR), "recording of " + (buf.length / SR).toFixed(1) + " s"); return; }
  try { LAB.cap = await capture(); b.textContent = "Stop recording"; b.classList.add("rec"); $("#labMsg").textContent = "Say Hey Google once, clearly, then press stop."; }
  catch (e) { LAB.cap = null; $("#labMsg").textContent = "Microphone unavailable: " + e.message; }
});
$("#labFile").addEventListener("change", ev => { const f = ev.target.files[0]; if (f) labReady(f, f.name); ev.target.value = ""; });
$("#labGo").addEventListener("click", async () => {
  if (!LAB.blob) return;
  const fd = new FormData(); fd.append("audio", LAB.blob, "input.wav"); fd.append("n", $("#labN").value);
  $("#labMsg").textContent = "Generating"; $("#labGo").disabled = true;
  try {
    const d = await api("/api/synthesize", { method: "POST", body: fd });
    const out = $("#labOut"); out.innerHTML = "";
    for (const it of d.items) {
      const wav = b64ToBlob(it.wav_b64, "audio/wav");
      const div = el("div", "labItem");
      const head = el("div"); head.append(el("b", "", it.type.replace(/_/g, " ")), el("span", "hint", "  " + it.duration.toFixed(2) + " s")); div.appendChild(head);
      const cv = el("canvas"); cv.width = 520; cv.height = 110; div.appendChild(cv);
      const ev = el("div", "tags"); it.events.forEach(e => ev.appendChild(eventTag(e))); div.appendChild(ev);
      const au = el("audio"); au.controls = true; au.src = URL.createObjectURL(wav); div.appendChild(au);
      out.appendChild(div);
      decodeBlob(wav).then(({ samples, rate }) => drawLab(cv, samples, rate, it.segments)).catch(() => { });
    }
    $("#labMsg").textContent = "Done: " + d.items.length + " clips. Teal is Hey, blue is Goo, amber is gle, grey is pause.";
  } catch (e) { $("#labMsg").textContent = "Error: " + e.message; }
  $("#labGo").disabled = false;
});
function drawLab(cv, samples, rate, segs) {
  const ctx = cv.getContext("2d"), W = cv.width, dur = samples.length / rate, X = t => (t / dur) * W, C = palette();
  const step = Math.max(1, Math.floor(samples.length / W)); ctx.fillStyle = cssVar("--ink-muted");
  for (let x = 0; x < W; x++) {
    let mn = 1, mx = -1; for (let j = x * step; j < Math.min(samples.length, (x + 1) * step); j++) { const v = samples[j]; if (v < mn) mn = v; if (v > mx) mx = v; }
    if (mx >= mn) ctx.fillRect(x, 40 - Math.min(mx / 0.9, 1) * 40, 1, Math.max(1, (mx - mn) / 0.9 * 40));
  }
  segs.forEach(([s, e, c]) => { ctx.fillStyle = C[CLASSES[c]] || cssVar("--line"); ctx.fillRect(X(s), 88, Math.max(1, X(e) - X(s)), 16); });
}

/* ------------------------------------------------------------------ evaluation */
function simpleTable(title, obj, key, label) {
  const wrap = el("div", "card"); wrap.appendChild(el("h2", "", title));
  const t = el("table"); const th = el("thead"); const hr = el("tr");
  [["", ""], ["Clips", "num"], [label, "num"]].forEach(([h, c]) => hr.appendChild(el("th", c, h))); th.appendChild(hr); t.appendChild(th);
  const tb = el("tbody");
  Object.entries(obj || {}).forEach(([k, v]) => { const r = el("tr"); r.append(el("td", "", k), el("td", "num", String(v.n)), el("td", "num", pct(v[key]))); tb.appendChild(r); });
  t.appendChild(tb); const w = el("div", "tableWrap"); w.appendChild(t); wrap.appendChild(w); return wrap;
}
function metric(name, val, def) {
  const c = el("div", "card metric"); c.append(el("div", "name", name), el("div", "val", val), el("div", "def", def)); return c;
}
async function loadReport() {
  const box = $("#reportBox");
  let m;
  try { m = await api("/api/model/report"); }
  catch (e) {
    box.innerHTML = ""; const c = el("div", "card");
    c.appendChild(el("p", "empty", "Evaluation results are not available on this server."));
    c.appendChild(el("p", "hint", "Server message: " + e.message)); box.appendChild(c); return;
  }
  box.innerHTML = "";
  const meta = el("p", "hint", "Split: " + m.split + "   Threshold: " + num(m.threshold) + "   Wake word clips: " + m.n_wake_clips + "   Negative clips: " + m.n_negative_clips + " (" + m.negative_hours + " h)   AUC: " + m.AUC);
  meta.style.marginBottom = "16px"; box.appendChild(meta);
  if (m.report_older_than_model) {
    const w = el("div", "card"); w.appendChild(el("p", "msg-error", "This report (" + m.report_time + ") describes an earlier model than the one loaded now (" + m.model_time + "). The Check tab shows results for the current model."));
    box.appendChild(w);
  }

  const grid = el("div", "metrics");
  grid.appendChild(metric("False accept rate", pct(m.FAR_overall, 2), "Share of negative clips wrongly accepted as the wake word."));
  grid.appendChild(metric("False reject rate", pct(m.FRR_overall), "Share of wake word clips the detector missed."));
  grid.appendChild(metric("False accepts per hour", num(m.false_accepts_per_hour), "False accepts divided by hours of negative audio in this split."));
  grid.appendChild(metric("False reject rate, fluent", pct(m.FRR_fluent), "Missed wake word clips among fluent clips."));
  grid.appendChild(metric("False reject rate, disfluent", pct(m.FRR_disfluent), "Missed wake word clips among disfluent clips."));
  box.appendChild(grid);

  const nW = m.n_wake_clips, nN = m.n_negative_clips, fn = Math.round(nW * m.FRR_overall), tp = nW - fn, fp = m.false_accepts, tn = nN - fp;
  const cmC = el("div", "card"); cmC.appendChild(el("h2", "", "Confusion matrix"));
  const cm = el("div", "cm");
  const cell = (n, l) => { const d = el("div"); d.append(el("div", "n", String(n)), el("div", "l", l)); return d; };
  const hd = t => el("div", "h", t);
  cm.append(hd(""), hd("Predicted wake word"), hd("Predicted not wake word"),
    hd("Actual wake word"), cell(tp, "accepted"), cell(fn, "missed"),
    hd("Actual negative"), cell(fp, "false accept"), cell(tn, "rejected"));
  cmC.appendChild(cm);
  cmC.appendChild(el("p", "hint cmap", "Negative counts come from the report. Wake word counts are rebuilt from the false reject rate and the clip count, so they can differ by one from rounding."));
  box.appendChild(cmC);

  box.appendChild(simpleTable("Missed wake words by disfluency type", m.FRR_by_disfluency_type, "FRR", "False reject rate"));
  box.appendChild(simpleTable("False accepts by negative category", m.FAR_by_negative_category, "FAR", "False accept rate"));
  box.appendChild(simpleTable("Missed wake words by speaker kind", m.FRR_by_speaker_kind, "FRR", "False reject rate"));
  [["FRR_by_stutter_profile", "Missed wake words by stuttering profile"], ["FRR_by_emotion", "Missed wake words by emotion"],
   ["FRR_by_age_group", "Missed wake words by age group"], ["FRR_by_secondary_behaviour", "Missed wake words by secondary behaviour"]]
    .forEach(([k, title]) => { if (m[k] && Object.keys(m[k]).length) box.appendChild(simpleTable(title, m[k], "FRR", "False reject rate")); });

  try {
    const c = await api("/api/model/comparison");
    const card = el("div", "card"); card.appendChild(el("h2", "", "Model comparison. Selected: " + c.selected));
    card.appendChild(el("p", "hint", c.selection_rule));
    const t = el("table"); const th = el("thead"), hr = el("tr");
    [["Model", ""], ["Status", ""], ["Val TPR (worst of fluent/disfluent)", "num"], ["Val TPR fluent", "num"], ["Val TPR disfluent", "num"], ["Val AUC", "num"], ["Test FRR", "num"], ["Test FAR", "num"]].forEach(([h, k]) => hr.appendChild(el("th", k, h)));
    th.appendChild(hr); t.appendChild(th); const tb = el("tbody");
    c.candidates.forEach(r => {
      const tr = el("tr"); const n = el("td", "", r.name + (r.name === c.selected ? "  (selected)" : ""));
      tr.append(n, el("td", "", r.status || ""), el("td", "num", pct(r.val_TPR_at_FAR)), el("td", "num", pct(r.val_TPR_fluent)), el("td", "num", pct(r.val_TPR_disfluent)), el("td", "num", r.val_AUC === undefined || r.val_AUC === null ? "-" : String(r.val_AUC)), el("td", "num", pct(r.test_FRR)), el("td", "num", pct(r.test_FAR, 2)));
      tb.appendChild(tr);
    });
    t.appendChild(tb); const w = el("div", "tableWrap"); w.appendChild(t); card.appendChild(w); box.appendChild(card);
  } catch (e) { /* no comparison yet */ }
}
