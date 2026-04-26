import useApi from "../hooks/useApi";
import { Loading, ErrorState } from "../components/LoadingState";
import {
  ScatterChart, Scatter, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, Cell, ReferenceLine,
} from "recharts";

export default function Experiment2() {
  const { data, loading, error } = useApi("/api/experiments/2");

  if (loading) return <Loading />;
  if (error) return <ErrorState message={error} />;

  const { summary, roc_curve, coefficient_distribution, top_latents } = data;

  const pathogenLatents = top_latents.filter((l) => l.direction === "pathogen");
  const nonpathogenLatents = top_latents.filter((l) => l.direction === "nonpathogen");

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl mb-1">Linear Probe -- Pathogen Detection</h2>
        <p className="text-sm text-[#828282]" style={{ fontFamily: "'Roboto Condensed', sans-serif" }}>
          Logistic regression on SAE features to predict pathogen vs non-pathogen. Identifies which latents carry the most predictive weight.
        </p>
      </div>

      {/* Summary stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {[
          { label: "Accuracy", value: (summary.accuracy * 100).toFixed(1) + "%", color: "#4e8c02" },
          { label: "MCC", value: summary.mcc.toFixed(3), color: "#0d8ba1" },
          { label: "AUROC", value: summary.auroc.toFixed(3), color: "#8a0038" },
          { label: "Best C", value: summary.best_C, color: "#828282" },
        ].map((s) => (
          <div key={s.label} className="bg-white rounded-lg border border-gray-100 p-4 text-center shadow-sm">
            <div className="text-2xl" style={{ fontFamily: "'VT323', monospace", color: s.color }}>{s.value}</div>
            <div className="text-xs text-[#828282]">{s.label}</div>
          </div>
        ))}
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        {/* ROC Curve */}
        <div className="bg-white rounded-lg border border-gray-100 p-5 shadow-sm">
          <h3 className="text-sm font-semibold text-gray-700 mb-3" style={{ fontFamily: "'VT323', monospace", textTransform: "uppercase", fontSize: "1rem" }}>
            ROC Curve
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart margin={{ top: 10, right: 20, bottom: 30, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis type="number" dataKey="fpr" domain={[0, 1]} tick={{ fontSize: 11 }}
                label={{ value: "False Positive Rate", position: "bottom", offset: 15, style: { fontSize: 12 } }} />
              <YAxis type="number" dataKey="tpr" domain={[0, 1]} tick={{ fontSize: 11 }}
                label={{ value: "True Positive Rate", angle: -90, position: "insideLeft", offset: 10, style: { fontSize: 12 } }} />
              <Tooltip formatter={(v) => v.toFixed(3)} />
              <Scatter data={roc_curve} line={{ stroke: "#0d8ba1", strokeWidth: 2 }} fill="none" legendType="none" />
              <Scatter data={[{ fpr: 0, tpr: 0 }, { fpr: 1, tpr: 1 }]} line={{ stroke: "#d4d4d4", strokeWidth: 1, strokeDasharray: "5 5" }} fill="none" legendType="none" />
            </ScatterChart>
          </ResponsiveContainer>
          <p className="text-xs text-center text-[#828282] mt-1">AUROC = {summary.auroc.toFixed(3)}</p>
        </div>

        {/* Coefficient distribution */}
        <div className="bg-white rounded-lg border border-gray-100 p-5 shadow-sm">
          <h3 className="text-sm font-semibold text-gray-700 mb-3" style={{ fontFamily: "'VT323', monospace", textTransform: "uppercase", fontSize: "1rem" }}>
            Coefficient Distribution
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={coefficient_distribution}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="bin_center" tick={{ fontSize: 10 }} label={{ value: "Probe Coefficient", position: "bottom", offset: 10, style: { fontSize: 12 } }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <ReferenceLine x={0} stroke="#000" strokeWidth={0.5} />
              <Bar dataKey="count" radius={[2, 2, 0, 0]}>
                {coefficient_distribution.map((entry, i) => (
                  <Cell key={i} fill={entry.bin_center > 0.5 ? "#8a0038" : entry.bin_center < -0.5 ? "#0d8ba1" : "#d4d4d4"} fillOpacity={0.7} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <p className="text-xs text-center text-[#828282] mt-1">32,768 latent weights. Most near zero; tails are informative.</p>
        </div>
      </div>

      {/* Top latents table */}
      <div className="grid md:grid-cols-2 gap-6">
        <LatentTable title="Top Pathogen-Associated" latents={pathogenLatents} color="#8a0038" />
        <LatentTable title="Top Non-Pathogen-Associated" latents={nonpathogenLatents} color="#0d8ba1" />
      </div>
    </div>
  );
}

function LatentTable({ title, latents, color }) {
  return (
    <div className="bg-white rounded-lg border border-gray-100 p-5 shadow-sm">
      <h3 className="text-sm font-semibold text-gray-700 mb-3" style={{ fontFamily: "'VT323', monospace", textTransform: "uppercase", fontSize: "1rem" }}>
        {title}
      </h3>
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-gray-200 text-left text-xs text-[#828282]">
            <th className="py-2 pr-2">Latent</th>
            <th className="py-2 pr-2">Coeff</th>
            <th className="py-2 pr-2">Freq(P)</th>
            <th className="py-2 pr-2">Freq(NP)</th>
            <th className="py-2">Enrich</th>
          </tr>
        </thead>
        <tbody>
          {latents.map((l) => (
            <tr key={l.latent_id} className="border-b border-gray-50 hover:bg-gray-50">
              <td className="py-2 pr-2 font-mono text-xs">#{l.latent_id}</td>
              <td className="py-2 pr-2 font-bold" style={{ color }}>{l.coefficient.toFixed(2)}</td>
              <td className="py-2 pr-2">{(l.freq_pathogen * 100).toFixed(0)}%</td>
              <td className="py-2 pr-2">{(l.freq_nonpathogen * 100).toFixed(0)}%</td>
              <td className="py-2">{l.enrichment.toFixed(1)}x</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
