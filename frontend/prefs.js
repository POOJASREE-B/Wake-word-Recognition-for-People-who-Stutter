/* Preferences: stored in this browser only (localStorage), applied at once, no save button. */
"use strict";
const PREF_DEFAULTS = { theme: "system", size: "normal", sens: 0, tail: 0.3, voice: true, beep: true, slow: false };
const PREF_CHOICES = { theme: ["system", "light", "dark"], size: ["normal", "large"], sens: [-0.05, 0, 0.05], tail: [0.3, 0.6, 1.0] };
const PREFS = Object.assign({}, PREF_DEFAULTS);
(function loadPrefs() {
  try {
    const s = JSON.parse(localStorage.getItem("wwPrefs") || "{}");
    Object.keys(PREF_DEFAULTS).forEach(k => {
      if (!(k in s) || typeof s[k] !== typeof PREF_DEFAULTS[k]) return;
      if (PREF_CHOICES[k] && !PREF_CHOICES[k].includes(s[k])) return;
      PREFS[k] = s[k];
    });
    if (!("theme" in s)) { const old = localStorage.getItem("wwTheme"); if (old === "dark" || old === "light") PREFS.theme = old; }
  } catch (e) { /* storage unavailable: defaults */ }
})();
function savePrefs() { try { localStorage.setItem("wwPrefs", JSON.stringify(PREFS)); } catch (e) { /* ignore */ } }
function effectiveTheme() {
  if (PREFS.theme !== "system") return PREFS.theme;
  return window.matchMedia && matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}
function applyPrefs() {
  const r = document.documentElement;
  r.setAttribute("data-theme", effectiveTheme()); r.setAttribute("data-size", PREFS.size);
  if (typeof redrawForTheme === "function") redrawForTheme();
}
applyPrefs();
if (window.matchMedia) matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { if (PREFS.theme === "system") applyPrefs(); });

document.addEventListener("DOMContentLoaded", () => {
  const dlg = document.getElementById("prefs"), btn = document.getElementById("prefsBtn");
  if (!dlg || !btn) return;
  const radios = Array.from(dlg.querySelectorAll("input[type=radio]")), boxes = Array.from(dlg.querySelectorAll("input[type=checkbox]"));
  const sync = () => {
    radios.forEach(r => { r.checked = String(PREFS[r.name]) === r.value; });
    boxes.forEach(b => { b.checked = !!PREFS[b.name]; });
    const thr = (typeof APP !== "undefined" && APP.threshold != null) ? APP.threshold : null;
    document.getElementById("prefsThr").textContent = thr == null ? "The model threshold is not loaded yet." :
      "Model threshold " + thr.toFixed(2) + ", used on this page: " + (thr + PREFS.sens).toFixed(2) + ".";
  };
  btn.addEventListener("click", () => { sync(); dlg.showModal(); });
  dlg.addEventListener("click", e => { if (e.target === dlg) dlg.close(); });            // click outside closes
  dlg.addEventListener("change", e => {
    const t = e.target;
    if (t.type === "radio") PREFS[t.name] = typeof PREF_DEFAULTS[t.name] === "number" ? Number(t.value) : t.value;
    else if (t.type === "checkbox") PREFS[t.name] = t.checked;
    savePrefs(); applyPrefs(); sync();
  });
  document.getElementById("prefsReset").addEventListener("click", () => { Object.assign(PREFS, PREF_DEFAULTS); savePrefs(); applyPrefs(); sync(); });
});
