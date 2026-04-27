import useApi from "../hooks/useApi";
import { Loading, ErrorState } from "../components/LoadingState";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from "recharts";

const LAYER_COLORS = { 8: "#b88a00", 16: "#4e8c02", 24: "#6b4fa0", 32: "#0d8ba1" };

export default function SAEHealth() {
  const { data, loading, error } = useApi("/api/page/sae-health");

  if (loading) return <Loading />;
  if (error) return <ErrorState message={error} />;

  const { summary, sequences_per_latent, max_activation_dist, active_features_per_seq, comparison, probe_by_layer } = data;

  const probeComparison = Object.entries(probe_by_layer).map(([layer, metrics]) => ({
    layer: `L${layer}`,
    layerNum: parseInt(layer),
    auroc: metrics.auroc,
    accuracy: metrics.accuracy,
  })).sort((a, b) => a.layerNum - b.layerNum);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl mb-1">SAE Health Check</h2>
        <p className="text-sm text-[#828282]" style={{ fontFamily: "'Roboto Condensed', sans-serif" }}>
          Verifying the sparse autoencoder is well-trained: dead latent census, activation distributions, and downstream classification.
        </p>
      </div>

      {/* Summary stats */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        {[
          { label: "Total Latents", value: summary.total_latents.toLocaleString(), color: "#0d8ba1" },
          { label: "Alive", value: summary.alive_count.toLocaleString(), color: "#4e8c02" },
          { label: "Dead", value: summary.dead_count.toLocaleString(), color: "#8a0038" },
          { label: "Dead %", value: summary.dead_pct.toFixed(2) + "%", color: "#b88a00" },
          { label: "Active / Sequence", value: summary.mean_active_per_seq, color: "#6b4fa0" },
        ].map((s) => (
          <div key={s.label} className="bg-white rounded-lg border border-gray-100 p-4 text-center shadow-sm">
            <div className="text-2xl" style={{ fontFamily: "'VT323', monospace", color: s.color }}>{s.value}</div>
            <div className="text-xs text-[#828282]">{s.label}</div>
          </div>
        ))}
      </div>

      {/* Histograms */}
      <div className="grid md:grid-cols-2 gap-6">
        <HistogramCard
          title="Sequences Per Latent"
          subtitle="How many sequences activate each latent (long-tail expected)"
          data={sequences_per_latent}
          xKey="bin_start"
          yKey="count"
          xLabel="# Sequences"
          color="#0d8ba1"
        />
        <HistogramCard
          title="Max Activation Distribution"
          subtitle="Peak activation value across all sequences for each latent"
          data={max_activation_dist}
          xKey="bin_start"
          yKey="count"
          xLabel="Max Activation"
          color="#4e8c02"
        />
      </div>

      <HistogramCard
        title="Active Features Per Sequence"
        subtitle="Number of latents firing per sequence (aggregated across tokens; per-token TopK = 64)"
        data={active_features_per_seq}
        xKey="bin_center"
        yKey="count"
        xLabel="# Active Features"
        color="#8a0038"
      />

      {/* Probe classification across layers */}
      <div className="bg-white rounded-lg border border-gray-100 p-5 shadow-sm">
        <h3 className="text-sm font-semibold text-gray-700 mb-1" style={{ fontFamily: "'VT323', monospace", textTransform: "uppercase", fontSize: "1rem" }}>
          Linear Probe Classification (Per Layer)
        </h3>
        <p className="text-xs text-[#828282] mb-4" style={{ fontFamily: "'Roboto Condensed', sans-serif" }}>
          SAE features at every layer support near-perfect pathogen classification. This confirms the SAE captures biologically meaningful signal.
        </p>
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
                  <Cell key={entry.layerNum} fill={LAYER_COLORS[entry.layerNum]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <div>
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200 text-left text-xs text-[#828282]">
                  <th className="py-2 pr-3">Layer</th>
                  <th className="py-2 pr-3 text-right">AUROC</th>
                  <th className="py-2 text-right">Accuracy</th>
                </tr>
              </thead>
              <tbody>
                {probeComparison.map((row) => (
                  <tr key={row.layerNum} className="border-b border-gray-50">
                    <td className="py-2.5 pr-3 font-medium" style={{ color: LAYER_COLORS[row.layerNum] }}>Layer {row.layerNum}</td>
                    <td className="py-2.5 pr-3 text-right font-mono">{row.auroc?.toFixed(4)}</td>
                    <td className="py-2.5 text-right font-mono">{(row.accuracy * 100)?.toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Comparison with InterProt */}
      {comparison && (
        <div className="bg-white rounded-lg border border-gray-100 p-5 shadow-sm">
          <h3 className="text-sm font-semibold text-gray-700 mb-3" style={{ fontFamily: "'VT323', monospace", textTransform: "uppercase", fontSize: "1rem" }}>
            Comparison with InterProt
          </h3>
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200 text-left text-xs text-[#828282]">
                <th className="py-2 pr-4">Property</th>
                <th className="py-2 pr-4 text-right">InterProt (ESM-2)</th>
                <th className="py-2 text-right">Ours (MetaGene-1)</th>
              </tr>
            </thead>
            <tbody>
              {[
                { prop: "d_model", ip: comparison.interprot.d_model, ours: comparison.ours.d_model },
                { prop: "Expansion", ip: comparison.interprot.expansion, ours: "8x" },
                { prop: "TopK", ip: comparison.interprot.k, ours: comparison.ours.k },
                { prop: "Total Latents", ip: comparison.interprot.total_latents, ours: comparison.ours.total_latents.toLocaleString() },
                { prop: "Dead %", ip: comparison.interprot.dead_pct, ours: comparison.ours.dead_pct + "%" },
              ].map((row) => (
                <tr key={row.prop} className="border-b border-gray-50">
                  <td className="py-2 pr-4 font-medium">{row.prop}</td>
                  <td className="py-2 pr-4 text-right font-mono text-[#828282]">{row.ip}</td>
                  <td className="py-2 text-right font-mono text-[#0d8ba1]">{row.ours}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function HistogramCard({ title, subtitle, data, xKey, yKey, xLabel, color }) {
  return (
    <div className="bg-white rounded-lg border border-gray-100 p-5 shadow-sm">
      <h3 className="text-sm font-semibold text-gray-700 mb-1" style={{ fontFamily: "'VT323', monospace", textTransform: "uppercase", fontSize: "1rem" }}>
        {title}
      </h3>
      {subtitle && (
        <p className="text-xs text-[#828282] mb-3" style={{ fontFamily: "'Roboto Condensed', sans-serif" }}>{subtitle}</p>
      )}
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis dataKey={xKey} tick={{ fontSize: 10 }}
            label={{ value: xLabel, position: "bottom", offset: 10, style: { fontSize: 12 } }} />
          <YAxis tick={{ fontSize: 11 }} />
          <Tooltip />
          <Bar dataKey={yKey} fill={color} fillOpacity={0.7} radius={[2, 2, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
