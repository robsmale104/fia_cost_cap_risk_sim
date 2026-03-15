// FIA Cost Cap Risk Simulator — Interactive Dashboard
// Built with React + Recharts
// For GitHub: https://github.com/[username]/f1-cost-cap-simulator
 
import { useState, useEffect, useCallback } from "react";
import {
  AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine, Cell, LineChart, Line, Legend
} from "recharts";
 
// ─── Monte Carlo Engine (JS port of simulation.py) ───────────────────────────
 
const COST_CAP = 135.0;
 
const DEFAULT_CATEGORIES = [
  { name: "Race Operations",          baseline: 28, low: 25, high: 34, dist: "normal",    color: "#E8002D" },
  { name: "Aerodynamic Development",  baseline: 22, low: 18, high: 30, dist: "triangular",color: "#FF8700" },
  { name: "Chassis & Bodywork",       baseline: 18, low: 15, high: 24, dist: "triangular",color: "#FFC906" },
  { name: "Power Unit (Leased)",      baseline: 15, low: 14, high: 17, dist: "normal",    color: "#00D2BE" },
  { name: "Manufacturing & Tooling",  baseline: 14, low: 11, high: 18, dist: "triangular",color: "#0067FF" },
  { name: "Transmission & Suspension",baseline: 12, low: 10, high: 16, dist: "triangular",color: "#9B59B6" },
  { name: "Electronics & Software",   baseline: 10, low: 8.5,high: 13, dist: "triangular",color: "#2ECC71" },
  { name: "Contingency & Risk",       baseline:  8, low:  4, high: 14, dist: "pert",      color: "#95A5A6" },
];
 
function seededRandom(seed) {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    return (s >>> 0) / 0xffffffff;
  };
}
 
function triangular(low, mode, high, rand) {
  const u = rand();
  const fc = (mode - low) / (high - low);
  if (u < fc) return low + Math.sqrt(u * (high - low) * (mode - low));
  return high - Math.sqrt((1 - u) * (high - low) * (high - mode));
}
 
function normalSample(mean, sigma, rand) {
  // Box-Muller
  const u1 = Math.max(rand(), 1e-10);
  const u2 = rand();
  return mean + sigma * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}
 
function pertSample(low, mode, high, rand) {
  const mean = (low + 4 * mode + high) / 6;
  const std = (high - low) / 6;
  const alpha = Math.max(0.5, ((mean - low) / (high - low)) * ((mean - low) * (high - mean) / (std * std) - 1));
  const beta = Math.max(0.5, alpha * (high - mean) / (mean - low));
  // Beta via normal approximation for simplicity
  const sigma = (high - low) / (6);
  return Math.max(low, Math.min(high, normalSample(mean, sigma, rand)));
}
 
function sampleCategory(cat, rand) {
  if (cat.dist === "normal") {
    const sigma = (cat.high - cat.low) / (2 * 1.645);
    return normalSample(cat.baseline, sigma, rand);
  }
  if (cat.dist === "pert") return pertSample(cat.low, cat.baseline, cat.high, rand);
  return triangular(cat.low, cat.baseline, cat.high, rand);
}
 
function runMonteCarlo(categories, N = 5000) {
  const rand = seededRandom(42);
  const totals = new Float64Array(N);
  const catSamples = categories.map(() => new Float64Array(N));
 
  for (let i = 0; i < N; i++) {
    let sum = 0;
    categories.forEach((cat, ci) => {
      const v = sampleCategory(cat, rand);
      catSamples[ci][i] = v;
      sum += v;
    });
    totals[i] = sum;
  }
 
  // Sort totals for CDF/histogram
  const sorted = Array.from(totals).sort((a, b) => a - b);
 
  // Breach stats
  const breachCount = sorted.filter(v => v > COST_CAP).length;
  const minorCount  = sorted.filter(v => v > COST_CAP && v <= COST_CAP * 1.05).length;
  const materialCount = sorted.filter(v => v > COST_CAP * 1.05).length;
  const excess = sorted.map(v => Math.max(0, v - COST_CAP));
  const expectedExcess = excess.reduce((a, b) => a + b, 0) / N;
 
  // Percentiles
  const pct = (p) => sorted[Math.floor(p / 100 * N)];
 
  // Histogram bins
  const min = pct(1), max = pct(99);
  const bins = 40;
  const width = (max - min) / bins;
  const hist = Array.from({ length: bins }, (_, i) => ({
    x: +(min + (i + 0.5) * width).toFixed(1),
    count: 0,
    overCap: min + (i + 0.5) * width > COST_CAP,
  }));
  sorted.forEach(v => {
    const bi = Math.min(bins - 1, Math.floor((v - min) / width));
    if (bi >= 0) hist[bi].count++;
  });
 
  // Sensitivity: correlation with total
  const totalMean = sorted.reduce((a, b) => a + b, 0) / N;
  const sensitivity = categories.map((cat, ci) => {
    const s = catSamples[ci];
    const sMean = Array.from(s).reduce((a, b) => a + b, 0) / N;
    let num = 0, dX = 0, dY = 0;
    for (let i = 0; i < N; i++) {
      num += (s[i] - sMean) * (totals[i] - totalMean);
      dX += (s[i] - sMean) ** 2;
      dY += (totals[i] - totalMean) ** 2;
    }
    return { name: cat.name, color: cat.color, corr: num / Math.sqrt(dX * dY) };
  }).sort((a, b) => b.corr - a.corr);
 
  // CDF data
  const cdfData = sorted.filter((_, i) => i % 50 === 0).map((v, i) => ({
    spend: +v.toFixed(1),
    probability: +((i * 50 / N) * 100).toFixed(1),
  }));
 
  return {
    hist,
    cdfData,
    sensitivity,
    breachProb: breachCount / N,
    minorBreachProb: minorCount / N,
    materialBreachProb: materialCount / N,
    expectedExcess,
    p10: pct(10), p25: pct(25), p50: pct(50), p75: pct(75), p90: pct(90),
    totalBaseline: categories.reduce((a, c) => a + c.baseline, 0),
  };
}
 
// ─── UI Components ────────────────────────────────────────────────────────────
 
const RISK_COLORS = { low: "#2ECC71", medium: "#FFC906", high: "#FF8700", critical: "#E8002D" };
 
function getRiskLevel(prob) {
  if (prob < 0.1) return "low";
  if (prob < 0.25) return "medium";
  if (prob < 0.5) return "high";
  return "critical";
}
 
function StatCard({ label, value, sub, risk }) {
  const col = risk ? RISK_COLORS[risk] : "#00D2BE";
  return (
    <div style={{
      background: "rgba(255,255,255,0.04)", border: `1px solid rgba(255,255,255,0.08)`,
      borderRadius: 12, padding: "20px 24px", borderTop: `3px solid ${col}`,
    }}>
      <div style={{ color: "#6B7280", fontSize: 11, letterSpacing: "0.12em", textTransform: "uppercase", marginBottom: 8 }}>{label}</div>
      <div style={{ color: col, fontSize: 28, fontWeight: 700, fontFamily: "monospace" }}>{value}</div>
      {sub && <div style={{ color: "#6B7280", fontSize: 12, marginTop: 4 }}>{sub}</div>}
    </div>
  );
}
 
function SliderRow({ cat, onChange }) {
  return (
    <div style={{ padding: "14px 0", borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
        <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ width: 10, height: 10, borderRadius: "50%", background: cat.color, display: "inline-block" }} />
          <span style={{ fontSize: 13, color: "#E5E7EB" }}>{cat.name}</span>
        </span>
        <span style={{ fontSize: 13, color: cat.color, fontFamily: "monospace", fontWeight: 600 }}>
          ${cat.baseline.toFixed(1)}M
        </span>
      </div>
      <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
        <span style={{ fontSize: 10, color: "#6B7280", width: 36 }}>${cat.low}M</span>
        <input
          type="range" min={cat.low} max={cat.high} step={0.5}
          value={cat.baseline}
          onChange={e => onChange(cat.name, "baseline", +e.target.value)}
          style={{ flex: 1, accentColor: cat.color }}
        />
        <span style={{ fontSize: 10, color: "#6B7280", width: 36 }}>${cat.high}M</span>
      </div>
    </div>
  );
}
 
const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: "#1a1a2e", border: "1px solid rgba(255,255,255,0.15)", borderRadius: 8, padding: "10px 14px" }}>
      <div style={{ color: "#9CA3AF", fontSize: 11 }}>Spend: ${label}M</div>
      <div style={{ color: payload[0].payload.overCap ? "#E8002D" : "#00D2BE", fontWeight: 600 }}>
        {payload[0].value} simulations
      </div>
    </div>
  );
};
 
// ─── Main App ─────────────────────────────────────────────────────────────────
 
export default function App() {
  const [categories, setCategories] = useState(DEFAULT_CATEGORIES);
  const [results, setResults] = useState(null);
  const [activeTab, setActiveTab] = useState("distribution");
 
  const recalculate = useCallback(() => {
    const res = runMonteCarlo(categories, 5000);
    setResults(res);
  }, [categories]);
 
  useEffect(() => { recalculate(); }, [recalculate]);
 
  const updateCat = (name, field, value) => {
    setCategories(prev => prev.map(c => c.name === name ? { ...c, [field]: value } : c));
  };
 
  const risk = results ? getRiskLevel(results.breachProb) : "low";
 
  const scenarios = [
    { label: "Conservative", mods: { "Race Operations": -3, "Aerodynamic Development": -4, "Contingency & Risk": -2 } },
    { label: "Baseline",     mods: {} },
    { label: "Aggressive",   mods: { "Race Operations": +4, "Aerodynamic Development": +5, "Contingency & Risk": +3 } },
  ];
 
  const scenarioData = scenarios.map(s => {
    const cats = categories.map(c => ({ ...c, baseline: c.baseline + (s.mods[c.name] || 0) }));
    const r = runMonteCarlo(cats, 3000);
    return { name: s.label, breach: +(r.breachProb * 100).toFixed(1), p50: +r.p50.toFixed(1), p90: +r.p90.toFixed(1) };
  });
 
  return (
    <div style={{
      minHeight: "100vh",
      background: "linear-gradient(135deg, #0a0a14 0%, #0d1117 50%, #0a0a14 100%)",
      color: "#E5E7EB",
      fontFamily: "'Segoe UI', system-ui, sans-serif",
      padding: "0 0 60px",
    }}>
 
      {/* Header */}
      <div style={{
        background: "rgba(255,255,255,0.02)",
        borderBottom: "1px solid rgba(255,255,255,0.08)",
        padding: "20px 40px",
        display: "flex", alignItems: "center", justifyContent: "space-between",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <div style={{
            width: 40, height: 40, borderRadius: 8,
            background: "linear-gradient(135deg, #E8002D, #FF8700)",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 18, fontWeight: 900,
          }}>F1</div>
          <div>
            <div style={{ fontSize: 18, fontWeight: 700, letterSpacing: "-0.01em" }}>
              FIA Cost Cap Risk Simulator
            </div>
            <div style={{ fontSize: 12, color: "#6B7280" }}>
              Monte Carlo Analysis · FIA Financial Regulations 2025 · $135M Cap
            </div>
          </div>
        </div>
        {results && (
          <div style={{
            background: `${RISK_COLORS[risk]}20`,
            border: `1px solid ${RISK_COLORS[risk]}40`,
            borderRadius: 20, padding: "6px 16px",
            color: RISK_COLORS[risk], fontSize: 13, fontWeight: 600,
          }}>
            {risk.toUpperCase()} BREACH RISK · {(results.breachProb * 100).toFixed(1)}%
          </div>
        )}
      </div>
 
      <div style={{ maxWidth: 1400, margin: "0 auto", padding: "32px 40px 0" }}>
 
        {/* KPI Row */}
        {results && (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 16, marginBottom: 32 }}>
            <StatCard label="Breach Probability" value={`${(results.breachProb * 100).toFixed(1)}%`}
              sub="P(Total > $135M)" risk={getRiskLevel(results.breachProb)} />
            <StatCard label="Minor Breach" value={`${(results.minorBreachProb * 100).toFixed(1)}%`}
              sub="0–5% over cap" risk={results.minorBreachProb > 0.1 ? "medium" : "low"} />
            <StatCard label="Material Breach" value={`${(results.materialBreachProb * 100).toFixed(1)}%`}
              sub=">5% over cap" risk={results.materialBreachProb > 0.05 ? "critical" : "low"} />
            <StatCard label="Median Spend" value={`$${results.p50.toFixed(1)}M`}
              sub={`P90: $${results.p90.toFixed(1)}M`} />
            <StatCard label="Expected Excess" value={`$${results.expectedExcess.toFixed(2)}M`}
              sub="Average overspend if breached" risk={results.expectedExcess > 2 ? "high" : results.expectedExcess > 0.5 ? "medium" : "low"} />
          </div>
        )}
 
        {/* Main Content */}
        <div style={{ display: "grid", gridTemplateColumns: "320px 1fr", gap: 24 }}>
 
          {/* Left — Controls */}
          <div style={{
            background: "rgba(255,255,255,0.03)",
            border: "1px solid rgba(255,255,255,0.08)",
            borderRadius: 16, padding: "24px",
          }}>
            <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 4 }}>Cost Category Controls</div>
            <div style={{ fontSize: 11, color: "#6B7280", marginBottom: 20 }}>
              Adjust baseline spend per category. Bounds define uncertainty range for Monte Carlo sampling.
            </div>
            {categories.map(cat => (
              <SliderRow key={cat.name} cat={cat} onChange={updateCat} />
            ))}
            {results && (
              <div style={{
                marginTop: 20, padding: "12px 16px",
                background: "rgba(255,255,255,0.04)", borderRadius: 8,
                display: "flex", justifyContent: "space-between",
              }}>
                <span style={{ fontSize: 12, color: "#9CA3AF" }}>Total Baseline</span>
                <span style={{ fontFamily: "monospace", fontWeight: 700, color: results.totalBaseline > COST_CAP ? "#E8002D" : "#2ECC71" }}>
                  ${results.totalBaseline.toFixed(1)}M / $135M
                </span>
              </div>
            )}
          </div>
 
          {/* Right — Charts */}
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
 
            {/* Tabs */}
            <div style={{ display: "flex", gap: 8 }}>
              {["distribution", "sensitivity", "scenarios"].map(tab => (
                <button key={tab} onClick={() => setActiveTab(tab)}
                  style={{
                    padding: "8px 20px", borderRadius: 8, border: "none", cursor: "pointer",
                    fontSize: 13, fontWeight: 500,
                    background: activeTab === tab ? "#E8002D" : "rgba(255,255,255,0.06)",
                    color: activeTab === tab ? "#fff" : "#9CA3AF",
                    transition: "all 0.2s",
                  }}>
                  {tab.charAt(0).toUpperCase() + tab.slice(1)}
                </button>
              ))}
            </div>
 
            {/* Distribution Tab */}
            {activeTab === "distribution" && results && (
              <div style={{
                background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.08)",
                borderRadius: 16, padding: 24,
              }}>
                <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 4 }}>
                  Spend Distribution — {results.hist.reduce((a, b) => a + b.count, 0).toLocaleString()} Simulations
                </div>
                <div style={{ fontSize: 11, color: "#6B7280", marginBottom: 20 }}>
                  Red bars indicate simulations where team exceeds the FIA cost cap
                </div>
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={results.hist} margin={{ top: 0, right: 0, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="x" tick={{ fill: "#6B7280", fontSize: 10 }}
                      tickFormatter={v => `$${v}M`} interval={7} />
                    <YAxis tick={{ fill: "#6B7280", fontSize: 10 }} />
                    <Tooltip content={<CustomTooltip />} />
                    <ReferenceLine x={COST_CAP} stroke="#E8002D" strokeWidth={2}
                      label={{ value: "CAP", fill: "#E8002D", fontSize: 11, position: "top" }} />
                    <Bar dataKey="count" radius={[2, 2, 0, 0]}>
                      {results.hist.map((entry, i) => (
                        <Cell key={i} fill={entry.overCap ? "#E8002D" : "#00D2BE"} fillOpacity={0.8} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
                {/* Percentile band */}
                <div style={{ display: "flex", justifyContent: "space-between", marginTop: 16, padding: "12px 16px", background: "rgba(0,0,0,0.2)", borderRadius: 8 }}>
                  {[["P10", results.p10], ["P25", results.p25], ["P50", results.p50], ["P75", results.p75], ["P90", results.p90]].map(([p, v]) => (
                    <div key={p} style={{ textAlign: "center" }}>
                      <div style={{ fontSize: 10, color: "#6B7280" }}>{p}</div>
                      <div style={{ fontSize: 14, fontWeight: 600, fontFamily: "monospace", color: v > COST_CAP ? "#E8002D" : "#E5E7EB" }}>
                        ${v.toFixed(1)}M
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
 
            {/* Sensitivity Tab */}
            {activeTab === "sensitivity" && results && (
              <div style={{
                background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.08)",
                borderRadius: 16, padding: 24,
              }}>
                <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 4 }}>Tornado Chart — Cost Driver Sensitivity</div>
                <div style={{ fontSize: 11, color: "#6B7280", marginBottom: 20 }}>
                  Pearson correlation of each category with total spend. Higher = greater influence on breach risk.
                </div>
                <ResponsiveContainer width="100%" height={320}>
                  <BarChart
                    data={results.sensitivity}
                    layout="vertical"
                    margin={{ top: 0, right: 40, left: 160, bottom: 0 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={false} />
                    <XAxis type="number" domain={[0, 0.7]} tick={{ fill: "#6B7280", fontSize: 10 }}
                      tickFormatter={v => v.toFixed(2)} />
                    <YAxis type="category" dataKey="name" tick={{ fill: "#E5E7EB", fontSize: 11 }} width={155} />
                    <Tooltip formatter={(v) => v.toFixed(3)} contentStyle={{ background: "#1a1a2e", border: "1px solid rgba(255,255,255,0.15)" }} />
                    <Bar dataKey="corr" radius={[0, 4, 4, 0]}>
                      {results.sensitivity.map((entry, i) => (
                        <Cell key={i} fill={entry.color} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
                <div style={{ marginTop: 16, padding: "12px 16px", background: "rgba(232,0,45,0.1)", borderRadius: 8, border: "1px solid rgba(232,0,45,0.2)" }}>
                  <span style={{ color: "#E8002D", fontWeight: 600, fontSize: 12 }}>Top risk driver: </span>
                  <span style={{ fontSize: 12, color: "#E5E7EB" }}>{results.sensitivity[0]?.name} — highest correlation with total spend. Focus cost control efforts here first.</span>
                </div>
              </div>
            )}
 
            {/* Scenarios Tab */}
            {activeTab === "scenarios" && (
              <div style={{
                background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.08)",
                borderRadius: 16, padding: 24,
              }}>
                <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 4 }}>Scenario Comparison</div>
                <div style={{ fontSize: 11, color: "#6B7280", marginBottom: 20 }}>
                  Conservative / Baseline / Aggressive development spend scenarios vs FIA cost cap
                </div>
                <ResponsiveContainer width="100%" height={240}>
                  <BarChart data={scenarioData} margin={{ top: 0, right: 20, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="name" tick={{ fill: "#E5E7EB", fontSize: 12 }} />
                    <YAxis tick={{ fill: "#6B7280", fontSize: 10 }} />
                    <Tooltip contentStyle={{ background: "#1a1a2e", border: "1px solid rgba(255,255,255,0.15)" }} />
                    <Legend wrapperStyle={{ fontSize: 12, color: "#9CA3AF" }} />
                    <ReferenceLine y={135} stroke="#E8002D" strokeDasharray="4 4" label={{ value: "Cap", fill: "#E8002D", fontSize: 10 }} />
                    <Bar dataKey="p50" name="Median Spend ($M)" fill="#00D2BE" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="p90" name="P90 Spend ($M)" fill="#FF8700" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12, marginTop: 16 }}>
                  {scenarioData.map(s => (
                    <div key={s.name} style={{
                      padding: "14px", borderRadius: 10,
                      background: "rgba(255,255,255,0.04)",
                      border: `1px solid ${s.breach > 30 ? "rgba(232,0,45,0.3)" : s.breach > 10 ? "rgba(255,135,0,0.3)" : "rgba(46,204,113,0.3)"}`,
                    }}>
                      <div style={{ fontSize: 12, color: "#9CA3AF", marginBottom: 6 }}>{s.name}</div>
                      <div style={{ fontSize: 22, fontWeight: 700, fontFamily: "monospace",
                        color: s.breach > 30 ? "#E8002D" : s.breach > 10 ? "#FF8700" : "#2ECC71" }}>
                        {s.breach}%
                      </div>
                      <div style={{ fontSize: 11, color: "#6B7280" }}>breach probability</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
 
        {/* Footer */}
        <div style={{ marginTop: 32, padding: "20px 0", borderTop: "1px solid rgba(255,255,255,0.06)", color: "#4B5563", fontSize: 11 }}>
          Built as part of F1 cost analysis portfolio · FIA Financial Regulations Article 4 · 5,000 Monte Carlo simulations per run ·
          Distributions: Triangular (default), Normal (fixed costs), PERT (contingency)
        </div>
      </div>
    </div>
  );
}