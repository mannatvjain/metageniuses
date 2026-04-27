import { useState, useRef } from "react";
import useApi from "../hooks/useApi";
import { Loading, ErrorState } from "../components/LoadingState";
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, Cell, PieChart, Pie,
} from "recharts";

const COLORS = { pathogen: "#8a0038", nonpathogen: "#0d8ba1", ns: "#d4d4d4" };
const CONF_COLORS = { high: "#4e8c02", medium: "#b88a00", low: "#828282" };

export default function Experiment1() {
  const { data, loading, error } = useApi("/api/experiments/1");
  const [selectedLatent, setSelectedLatent] = useState(null);
  const [filter, setFilter] = useState("all");
  const detailRef = useRef(null);

  if (loading) return <Loading />;
  if (error) return <ErrorState message={error} />;

  const { summary, volcano, detectors, enrichment_histogram } = data;

  const filteredDetectors = detectors.filter((d) => {
    if (filter === "all") return true;
    return d.confidence === filter;
  });

  const selected = detectors.find((d) => d.latent_id === selectedLatent);

  const handleSelectLatent = (latentId) => {
    setSelectedLatent(latentId);
    setTimeout(() => detailRef.current?.scrollIntoView({ behavior: "smooth" }), 100);
  };

  return (
    <div className="space-y-6">
      {/* Title */}
      <div>
        <h2 className="text-2xl mb-1">Pathogen Detector Latents</h2>
        <p className="text-sm text-[#828282]" style={{ fontFamily: "'Roboto Condensed', sans-serif" }}>
          SAE latents that fire specifically on pathogen sequences, identified via enrichment analysis and validated with NCBI BLAST.
        </p>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-2 md:grid-cols-6 gap-3">
        {[
          { label: "Total Latents", value: summary.total_latents?.toLocaleString() },
          { label: "Pathogen-Enriched", value: summary.pathogen_enriched?.toLocaleString() },
          { label: "Pathogen-Specific", value: summary.pathogen_specific },
          { label: "High Confidence", value: summary.high_confidence_detectors, color: CONF_COLORS.high },
          { label: "Medium Confidence", value: summary.medium_confidence_detectors, color: CONF_COLORS.medium },
          { label: "Non-Pathogen-Enriched", value: summary.nonpathogen_enriched?.toLocaleString() },
        ].map((s) => (
          <div key={s.label} className="bg-white rounded-lg border border-gray-100 p-3 text-center shadow-sm">
            <div className="text-xl" style={{ fontFamily: "'VT323', monospace", color: s.color || "#0d8ba1" }}>{s.value}</div>
            <div className="text-xs text-[#828282]">{s.label}</div>
          </div>
        ))}
      </div>

      {/* Volcano plot */}
      <div className="bg-white rounded-lg border border-gray-100 p-5 shadow-sm">
        <h3 className="text-sm font-semibold text-gray-700 mb-3" style={{ fontFamily: "'VT323', monospace", textTransform: "uppercase", fontSize: "1rem" }}>
          Volcano Plot
        </h3>
        <p className="text-xs text-[#828282] mb-3" style={{ fontFamily: "'Roboto Condensed', sans-serif" }}>
          Each point is one SAE latent. Click colored points to inspect organism detectors.
        </p>
        <ResponsiveContainer width="100%" height={450}>
          <ScatterChart margin={{ top: 10, right: 20, bottom: 30, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis type="number" dataKey="log2fc" name="log2FC" domain={[-6, 6]}
              label={{ value: "log2 Fold Change", position: "bottom", offset: 15, style: { fontSize: 12 } }} />
            <YAxis type="number" dataKey="neg_log10_pval" name="-log10(p)"
              label={{ value: "-log10(FDR p-value)", angle: -90, position: "insideLeft", offset: 10, style: { fontSize: 12 } }} />
            <Tooltip content={<VolcanoTooltip />} />
            {["ns", "nonpathogen", "pathogen"].map((dir) => (
              <Scatter
                key={dir}
                data={volcano.filter((v) => v.direction === dir)}
                fill={COLORS[dir]}
                fillOpacity={dir === "ns" ? 0.2 : 0.7}
                r={dir === "ns" ? 1.5 : 3}
                onClick={(e) => {
                  if (e && e.latent_id != null) handleSelectLatent(e.latent_id);
                }}
                cursor={dir !== "ns" ? "pointer" : "default"}
              />
            ))}
          </ScatterChart>
        </ResponsiveContainer>
        <div className="flex items-center justify-center gap-6 mt-2 text-xs text-[#828282]">
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-[#8a0038] inline-block" />Pathogen-enriched</span>
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-[#0d8ba1] inline-block" />Non-pathogen-enriched</span>
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-[#d4d4d4] inline-block" />Not significant</span>
        </div>
      </div>

      {/* Organism detector table */}
      <div className="bg-white rounded-lg border border-gray-100 p-5 shadow-sm">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold text-gray-700" style={{ fontFamily: "'VT323', monospace", textTransform: "uppercase", fontSize: "1rem" }}>
            Organism Detectors ({filteredDetectors.length})
          </h3>
          <div className="flex gap-2">
            {["all", "high", "medium", "low"].map((f) => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={`px-2.5 py-1 rounded text-xs font-medium transition-colors ${
                  filter === f
                    ? "bg-[#0d8ba1] text-white"
                    : "bg-gray-100 text-gray-600 hover:bg-gray-200"
                }`}
              >
                {f === "all" ? "All" : f.charAt(0).toUpperCase() + f.slice(1)}
              </button>
            ))}
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200 text-left text-xs text-[#828282]">
                <th className="py-2 pr-3">Latent</th>
                <th className="py-2 pr-3">Organism</th>
                <th className="py-2 pr-3">Confidence</th>
                <th className="py-2 pr-3">Consistency</th>
                <th className="py-2 pr-3">OR</th>
                <th className="py-2 pr-3">F1</th>
                <th className="py-2 pr-3">Identity</th>
              </tr>
            </thead>
            <tbody>
              {filteredDetectors.map((d) => (
                <tr
                  key={d.latent_id}
                  onClick={() => handleSelectLatent(d.latent_id)}
                  className={`border-b border-gray-50 cursor-pointer transition-colors ${
                    selectedLatent === d.latent_id
                      ? "bg-[#0d8ba1]/10"
                      : "hover:bg-gray-50"
                  }`}
                >
                  <td className="py-2 pr-3 font-mono text-xs">#{d.latent_id}</td>
                  <td className="py-2 pr-3 font-bold text-[#8a0038]">{d.dominant_organism || "—"}</td>
                  <td className="py-2 pr-3">
                    <span
                      className="px-1.5 py-0.5 rounded text-xs font-bold"
                      style={{
                        backgroundColor: (CONF_COLORS[d.confidence] || "#828282") + "20",
                        color: CONF_COLORS[d.confidence] || "#828282",
                      }}
                    >
                      {d.confidence}
                    </span>
                  </td>
                  <td className="py-2 pr-3 font-mono text-xs">{d.hit_consistency}</td>
                  <td className="py-2 pr-3">{d.fisher_or === 999 ? "∞" : d.fisher_or?.toFixed(1)}</td>
                  <td className="py-2 pr-3">{d.best_f1?.toFixed(3)}</td>
                  <td className="py-2 pr-3">{d.mean_percent_identity?.toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Detail panel */}
      {selected && (
        <div ref={detailRef} className="bg-white rounded-lg border-2 border-[#0d8ba1] p-5 shadow-sm">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-bold" style={{ fontFamily: "'VT323', monospace" }}>
              Latent #{selected.latent_id}: {selected.dominant_organism || "Unknown"}
            </h3>
            <button
              onClick={() => setSelectedLatent(null)}
              className="text-gray-400 hover:text-gray-600 text-sm"
            >
              Close
            </button>
          </div>

          {/* Stats row */}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-5">
            {[
              { label: "Confidence", value: selected.confidence, color: CONF_COLORS[selected.confidence] },
              { label: "Consistency", value: selected.hit_consistency },
              { label: "Odds Ratio", value: selected.fisher_or === 999 ? "∞" : selected.fisher_or },
              { label: "Best F1", value: selected.best_f1?.toFixed(3) },
              { label: "Mean Identity", value: selected.mean_percent_identity?.toFixed(1) + "%" },
            ].map((s) => (
              <div key={s.label} className="bg-gray-50 rounded p-2 text-center">
                <div className="text-lg font-bold" style={{ color: s.color || "#0d8ba1", fontFamily: "'VT323', monospace" }}>{s.value}</div>
                <div className="text-xs text-[#828282]">{s.label}</div>
              </div>
            ))}
          </div>

          {/* Organism distribution + BLAST hits */}
          <div className="grid md:grid-cols-3 gap-5">
            {/* Organism pie chart */}
            <div>
              <h4 className="text-xs font-semibold text-gray-500 mb-2 uppercase">Organism Distribution</h4>
              <OrganismPie hits={selected.blast_hits} />
            </div>

            {/* BLAST hits table */}
            <div className="md:col-span-2">
              <h4 className="text-xs font-semibold text-gray-500 mb-2 uppercase">BLAST Hits ({selected.blast_hits?.length || 0})</h4>
              <div className="overflow-x-auto max-h-80 overflow-y-auto">
                <table className="w-full text-xs">
                  <thead className="sticky top-0 bg-white">
                    <tr className="border-b border-gray-200 text-left text-[#828282]">
                      <th className="py-1.5 pr-2">Organism</th>
                      <th className="py-1.5 pr-2">Identity</th>
                      <th className="py-1.5 pr-2">E-value</th>
                      <th className="py-1.5 pr-2">Score</th>
                      <th className="py-1.5">Accession</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(selected.blast_hits || []).map((h, i) => (
                      <tr key={i} className="border-b border-gray-50 hover:bg-gray-50">
                        <td className="py-1.5 pr-2 font-medium text-[#8a0038]">{h.organism}</td>
                        <td className="py-1.5 pr-2 font-mono">{h.percent_identity?.toFixed(1)}%</td>
                        <td className="py-1.5 pr-2 font-mono">{h.e_value?.toExponential(1)}</td>
                        <td className="py-1.5 pr-2 font-mono">{h.bit_score?.toFixed(0)}</td>
                        <td className="py-1.5">
                          <a
                            href={`https://www.ncbi.nlm.nih.gov/nuccore/${h.accession}`}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-[#0d8ba1] hover:underline font-mono"
                          >
                            {h.accession}
                          </a>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Sequences */}
          {selected.blast_hits?.some((h) => h.sequence) && (
            <div className="mt-4">
              <h4 className="text-xs font-semibold text-gray-500 mb-2 uppercase">Top Activating Sequences</h4>
              <div className="space-y-2 max-h-60 overflow-y-auto">
                {selected.blast_hits.filter((h) => h.sequence).map((h, i) => (
                  <div key={i} className="bg-gray-50 rounded p-2">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="font-mono text-xs text-[#828282]">{h.sequence_id}</span>
                      <span className="text-xs text-[#8a0038]">{h.organism}</span>
                      <span className="text-xs text-[#828282]">{h.percent_identity?.toFixed(1)}%</span>
                    </div>
                    <div className="font-mono text-xs text-gray-700 break-all leading-relaxed tracking-wider">
                      {h.sequence}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Enrichment histogram */}
      <div className="bg-white rounded-lg border border-gray-100 p-5 shadow-sm">
        <h3 className="text-sm font-semibold text-gray-700 mb-3" style={{ fontFamily: "'VT323', monospace", textTransform: "uppercase", fontSize: "1rem" }}>
          Enrichment Distribution
        </h3>
        <p className="text-xs text-[#828282] mb-3" style={{ fontFamily: "'Roboto Condensed', sans-serif" }}>
          Distribution of log2(Fisher Odds Ratio) across all alive latents.
        </p>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={enrichment_histogram}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis dataKey="bin_start" tick={{ fontSize: 10 }}
              label={{ value: "log2(Odds Ratio)", position: "bottom", offset: 10, style: { fontSize: 12 } }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip content={<HistogramTooltip />} />
            <Bar dataKey="count" radius={[2, 2, 0, 0]}>
              {enrichment_histogram.map((entry, i) => (
                <Cell key={i} fill={entry.bin_start > 0 ? "#8a0038" : entry.bin_end < 0 ? "#0d8ba1" : "#d4d4d4"} fillOpacity={0.7} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function VolcanoTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="bg-white border border-gray-200 rounded-md shadow-sm p-2 text-xs">
      <p className="font-bold">Latent #{d.latent_id}</p>
      <p>log2FC: {d.log2fc?.toFixed(3)}</p>
      <p>-log10(p): {d.neg_log10_pval?.toFixed(2)}</p>
      <p className="capitalize">{d.direction}</p>
      {d.organism && <p className="font-bold text-[#8a0038] mt-1">{d.organism}</p>}
    </div>
  );
}

function HistogramTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="bg-white border border-gray-200 rounded-md shadow-sm p-2 text-xs">
      <p>log2(OR): {d.bin_start?.toFixed(2)} - {d.bin_end?.toFixed(2)}</p>
      <p>{d.count} latents</p>
    </div>
  );
}

const PIE_COLORS = ["#8a0038", "#0d8ba1", "#4e8c02", "#b88a00", "#6b4fa0", "#cc5500", "#2d7d9a", "#a0522d"];

function OrganismPie({ hits }) {
  if (!hits?.length) return <p className="text-xs text-[#828282]">No hits</p>;

  const counts = {};
  hits.forEach((h) => {
    const org = h.organism?.split(" ").slice(0, 2).join(" ") || "Unknown";
    counts[org] = (counts[org] || 0) + 1;
  });

  const pieData = Object.entries(counts)
    .map(([name, value]) => ({ name, value }))
    .sort((a, b) => b.value - a.value);

  return (
    <ResponsiveContainer width="100%" height={200}>
      <PieChart>
        <Pie
          data={pieData}
          dataKey="value"
          nameKey="name"
          cx="50%"
          cy="50%"
          outerRadius={70}
          label={({ name, value }) => `${name} (${value})`}
          labelLine={true}
        >
          {pieData.map((_, i) => (
            <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />
          ))}
        </Pie>
        <Tooltip />
      </PieChart>
    </ResponsiveContainer>
  );
}
