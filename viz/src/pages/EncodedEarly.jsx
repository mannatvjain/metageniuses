import { useState } from "react";
import useApi from "../hooks/useApi";
import { Loading, ErrorState } from "../components/LoadingState";
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, Cell, ReferenceLine,
} from "recharts";

const COLORS = { pathogen: "#8a0038", nonpathogen: "#0d8ba1", ns: "#d4d4d4" };
const LAYER_COLORS = { 8: "#b88a00", 16: "#4e8c02", 24: "#6b4fa0", 32: "#0d8ba1" };
const CONF_COLORS = { high: "#4e8c02", medium: "#b88a00", low: "#828282" };

export default function EncodedEarly() {
  const { data, loading, error } = useApi("/api/page/encoded-early");
  const [activeLayer, setActiveLayer] = useState(8);

  if (loading) return <Loading />;
  if (error) return <ErrorState message={error} />;

  const { probe_by_layer, enrichment_by_layer, volcano_by_layer, organism_labels_by_layer, layers } = data;
  const probe = probe_by_layer[activeLayer];
  const enrichment = enrichment_by_layer[activeLayer];
  const volcano = volcano_by_layer[activeLayer];
  const detectors = organism_labels_by_layer[activeLayer] || [];

  // Bar chart data for cross-layer probe comparison
  const probeComparison = layers.map((l) => ({
    layer: `L${l}`,
    layerNum: l,
    auroc: probe_by_layer[l]?.auroc,
    accuracy: probe_by_layer[l]?.accuracy,
  }));

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl mb-1">Pathogenicity Is Encoded Early</h2>
        <p className="text-sm text-[#828282]" style={{ fontFamily: "'Roboto Condensed', sans-serif" }}>
          Linear probes achieve near-identical AUROC at layers 8, 16, 24, and 32. Organism-specific detectors confirmed via BLAST at layers 16 and 32.
        </p>
      </div>

      {/* Cross-layer probe comparison — always visible */}
      <div className="bg-white rounded-lg border border-gray-100 p-5 shadow-sm">
        <h3 className="text-sm font-semibold text-gray-700 mb-3" style={{ fontFamily: "'VT323', monospace", textTransform: "uppercase", fontSize: "1rem" }}>
          Probe Performance Across Layers
        </h3>
        <div className="grid md:grid-cols-2 gap-6">
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={probeComparison} margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="layer" tick={{ fontSize: 12 }} />
              <YAxis domain={[0.93, 1.0]} tick={{ fontSize: 11 }}
                label={{ value: "AUROC", angle: -90, position: "insideLeft", offset: 10, style: { fontSize: 12 } }} />
              <Tooltip formatter={(v) => v?.toFixed(4)} />
              <Bar dataKey="auroc" radius={[4, 4, 0, 0]}>
                {probeComparison.map((entry) => (
                  <Cell
                    key={entry.layerNum}
                    fill={LAYER_COLORS[entry.layerNum]}
                    fillOpacity={entry.layerNum === activeLayer ? 1 : 0.35}
                    stroke={entry.layerNum === activeLayer ? "#000" : "none"}
                    strokeWidth={entry.layerNum === activeLayer ? 2 : 0}
                    cursor="pointer"
                    onClick={() => setActiveLayer(entry.layerNum)}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <div className="flex flex-col justify-center">
            <p className="text-sm text-[#828282] mb-4" style={{ fontFamily: "'Roboto Condensed', sans-serif" }}>
              Click a bar to explore that layer. Pathogen detection signal (AUROC 0.987-0.991) is present as early as layer 8 and maintained throughout the network.
            </p>
            <div className="grid grid-cols-4 gap-2">
              {layers.map((l) => (
                <button
                  key={l}
                  onClick={() => setActiveLayer(l)}
                  className={`px-3 py-2 rounded-lg text-sm font-medium transition-all ${
                    activeLayer === l
                      ? "text-white shadow-md scale-105"
                      : "bg-gray-100 text-gray-600 hover:bg-gray-200"
                  }`}
                  style={activeLayer === l ? { backgroundColor: LAYER_COLORS[l] } : {}}
                >
                  Layer {l}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Active layer detail section */}
      <div className="bg-white rounded-lg border-2 p-5 shadow-sm" style={{ borderColor: LAYER_COLORS[activeLayer] }}>
        <h3 className="text-lg font-bold mb-4" style={{ fontFamily: "'VT323', monospace", color: LAYER_COLORS[activeLayer] }}>
          Layer {activeLayer} Detail
        </h3>

        {/* Stats row */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-5">
          {[
            { label: "AUROC", value: probe?.auroc?.toFixed(4) },
            { label: "Accuracy", value: (probe?.accuracy * 100)?.toFixed(1) + "%" },
            { label: "Pathogen-Enriched", value: enrichment?.pathogen_enriched?.toLocaleString() },
            { label: "Pathogen-Specific", value: enrichment?.pathogen_specific },
            { label: "BLAST Detectors", value: detectors.length || "N/A" },
          ].map((s) => (
            <div key={s.label} className="bg-gray-50 rounded-lg p-3 text-center">
              <div className="text-xl" style={{ fontFamily: "'VT323', monospace", color: LAYER_COLORS[activeLayer] }}>{s.value}</div>
              <div className="text-xs text-[#828282]">{s.label}</div>
            </div>
          ))}
        </div>

        {/* Volcano plot */}
        <div className="mb-5">
          <h4 className="text-xs font-semibold text-gray-500 mb-2 uppercase">Enrichment Volcano</h4>
          <ResponsiveContainer width="100%" height={350}>
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
                  fillOpacity={dir === "ns" ? 0.15 : 0.6}
                  r={dir === "ns" ? 1 : 2.5}
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

        {/* Organism detectors table (only for layers with BLAST) */}
        {detectors.length > 0 ? (
          <div>
            <h4 className="text-xs font-semibold text-gray-500 mb-2 uppercase">
              BLAST-Validated Organism Detectors ({detectors.length})
            </h4>
            <div className="overflow-x-auto max-h-96 overflow-y-auto">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-white">
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
                  {detectors.map((d) => (
                    <tr key={d.latent_id} className="border-b border-gray-50 hover:bg-gray-50">
                      <td className="py-2 pr-3 font-mono text-xs">#{d.latent_id}</td>
                      <td className="py-2 pr-3 font-bold text-[#8a0038]">{d.dominant_organism}</td>
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
                      <td className="py-2 pr-3">{d.fisher_or === 999 ? "\u221E" : d.fisher_or?.toFixed(1)}</td>
                      <td className="py-2 pr-3">{d.best_f1?.toFixed(3)}</td>
                      <td className="py-2 pr-3">{d.mean_percent_identity?.toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ) : (
          <div className="bg-gray-50 rounded-lg p-4 text-center">
            <p className="text-sm text-[#828282]">
              BLAST validation not yet run for layer {activeLayer}. Enrichment data available above.
            </p>
          </div>
        )}
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
    </div>
  );
}
