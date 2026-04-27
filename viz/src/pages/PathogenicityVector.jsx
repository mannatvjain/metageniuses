import { useState } from "react";
import useApi from "../hooks/useApi";
import { Loading, ErrorState } from "../components/LoadingState";
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, Cell, ReferenceLine,
} from "recharts";

const COLORS = {
  pathogen: "#8a0038",
  nonpathogen: "#0d8ba1",
  positive: "#8a0038",
  negative: "#0d8ba1",
  both: "#8a0038",
  one: "#b88a00",
  ns: "#d4d4d4",
};

export default function PathogenicityVector() {
  const { data, loading, error } = useApi("/api/page/pathogenicity-vector");
  const [activeTab, setActiveTab] = useState("umap");

  if (loading) return <Loading />;
  if (error) return <ErrorState message={error} />;

  const { probe, roc_curve, top_latents, coefficient_distribution, umap, cross_delivery, stability_scatter } = data;

  const tabs = [
    { id: "umap", label: "Sequence UMAP" },
    { id: "probe", label: "Probe Weights" },
    { id: "generalization", label: "Cross-Delivery" },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl mb-1">The Pathogenicity Vector</h2>
        <p className="text-sm text-[#828282]" style={{ fontFamily: "'Roboto Condensed', sans-serif" }}>
          A linear probe defines a direction in SAE feature space that separates pathogen from non-pathogen.
          This direction generalizes across sequencing batches with only 0.8% accuracy drop.
        </p>
      </div>

      {/* Summary stats */}
      <div className="grid grid-cols-2 md:grid-cols-6 gap-3">
        {[
          { label: "Probe Accuracy", value: (probe.accuracy * 100).toFixed(1) + "%", color: "#4e8c02" },
          { label: "AUROC", value: probe.auroc.toFixed(3), color: "#0d8ba1" },
          { label: "MCC", value: probe.mcc.toFixed(3), color: "#8a0038" },
          { label: "Cross-Delivery Acc", value: (cross_delivery.class2_test.accuracy * 100).toFixed(1) + "%", color: "#4e8c02" },
          { label: "Delta", value: (cross_delivery.delta.accuracy * 100).toFixed(1) + "%", color: "#b88a00" },
          { label: "Feature Stability (rho)", value: cross_delivery.feature_stability.spearman_all.toFixed(3), color: "#6b4fa0" },
        ].map((s) => (
          <div key={s.label} className="bg-white rounded-lg border border-gray-100 p-3 text-center shadow-sm">
            <div className="text-xl" style={{ fontFamily: "'VT323', monospace", color: s.color }}>{s.value}</div>
            <div className="text-xs text-[#828282]">{s.label}</div>
          </div>
        ))}
      </div>

      {/* Tab selector */}
      <div className="flex gap-2">
        {tabs.map((t) => (
          <button
            key={t.id}
            onClick={() => setActiveTab(t.id)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              activeTab === t.id
                ? "bg-[#0d8ba1] text-white"
                : "bg-white border border-gray-200 text-gray-600 hover:bg-gray-50"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {activeTab === "umap" && <UmapPanel umap={umap} />}
      {activeTab === "probe" && (
        <ProbePanel
          roc_curve={roc_curve}
          top_latents={top_latents}
          coefficient_distribution={coefficient_distribution}
          probe={probe}
        />
      )}
      {activeTab === "generalization" && (
        <GeneralizationPanel
          cross_delivery={cross_delivery}
          stability_scatter={stability_scatter}
          roc_curve={roc_curve}
          cross_roc={data.cross_roc}
        />
      )}
    </div>
  );
}

function UmapPanel({ umap }) {
  const pathogen = umap.points.filter((p) => p.label === 1);
  const nonpathogen = umap.points.filter((p) => p.label === 0);

  return (
    <div className="bg-white rounded-lg border border-gray-100 p-5 shadow-sm">
      <h3 className="text-sm font-semibold text-gray-700 mb-1" style={{ fontFamily: "'VT323', monospace", textTransform: "uppercase", fontSize: "1rem" }}>
        Sequence UMAP
      </h3>
      <p className="text-xs text-[#828282] mb-3" style={{ fontFamily: "'Roboto Condensed', sans-serif" }}>
        20,000 sequences projected from 32,768-dim SAE space via PCA (50d) then UMAP. Hover for details.
      </p>
      <ResponsiveContainer width="100%" height={500}>
        <ScatterChart margin={{ top: 10, right: 20, bottom: 30, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis type="number" dataKey="x" name="UMAP 1" tick={{ fontSize: 11 }}
            label={{ value: "UMAP 1", position: "bottom", offset: 15, style: { fontSize: 12 } }} />
          <YAxis type="number" dataKey="y" name="UMAP 2" tick={{ fontSize: 11 }}
            label={{ value: "UMAP 2", angle: -90, position: "insideLeft", offset: 10, style: { fontSize: 12 } }} />
          <Tooltip content={<UmapTooltip />} />
          <Scatter name="Non-pathogen" data={nonpathogen} fill={COLORS.nonpathogen} fillOpacity={0.4} r={2} />
          <Scatter name="Pathogen" data={pathogen} fill={COLORS.pathogen} fillOpacity={0.4} r={2} />
        </ScatterChart>
      </ResponsiveContainer>
      <div className="flex items-center justify-center gap-6 mt-2 text-xs text-[#828282]">
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-[#8a0038] inline-block" />Pathogen ({pathogen.length.toLocaleString()})</span>
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-[#0d8ba1] inline-block" />Non-pathogen ({nonpathogen.length.toLocaleString()})</span>
      </div>
    </div>
  );
}

function UmapTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="bg-white border border-gray-200 rounded-md shadow-sm p-2 text-xs">
      <p className="font-bold">{d.sequence_id}</p>
      <p style={{ color: d.label === 1 ? COLORS.pathogen : COLORS.nonpathogen }}>
        {d.label === 1 ? "Pathogen" : "Non-pathogen"}
      </p>
    </div>
  );
}

function ProbePanel({ roc_curve, top_latents, coefficient_distribution, probe }) {
  const pathogenLatents = top_latents.filter((l) => l.direction === "pathogen");
  const nonpathogenLatents = top_latents.filter((l) => l.direction === "nonpathogen");

  return (
    <div className="space-y-5">
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
          <p className="text-xs text-center text-[#828282] mt-1">AUROC = {probe.auroc.toFixed(3)}</p>
        </div>

        {/* Coefficient distribution */}
        <div className="bg-white rounded-lg border border-gray-100 p-5 shadow-sm">
          <h3 className="text-sm font-semibold text-gray-700 mb-3" style={{ fontFamily: "'VT323', monospace", textTransform: "uppercase", fontSize: "1rem" }}>
            Coefficient Distribution
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={coefficient_distribution}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="bin_center" tick={{ fontSize: 10 }}
                label={{ value: "Probe Coefficient", position: "bottom", offset: 10, style: { fontSize: 12 } }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <ReferenceLine x={0} stroke="#000" strokeWidth={0.5} />
              <Bar dataKey="count" radius={[2, 2, 0, 0]}>
                {coefficient_distribution.map((entry, i) => (
                  <Cell key={i} fill={entry.bin_center > 0.5 ? COLORS.positive : entry.bin_center < -0.5 ? COLORS.negative : COLORS.ns} fillOpacity={0.7} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <p className="text-xs text-center text-[#828282] mt-1">32,768 latent weights. Tails are informative.</p>
        </div>
      </div>

      {/* Top latents tables */}
      <div className="grid md:grid-cols-2 gap-6">
        <LatentTable title="Top Pathogen-Associated" latents={pathogenLatents} color={COLORS.pathogen} />
        <LatentTable title="Top Non-Pathogen-Associated" latents={nonpathogenLatents} color={COLORS.nonpathogen} />
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

function GeneralizationPanel({ cross_delivery, stability_scatter, cross_roc }) {
  const { class1_train, class2_test, delta, feature_stability } = cross_delivery;

  const comparisonData = [
    { metric: "Accuracy", class1: class1_train.accuracy, class2: class2_test.accuracy },
    { metric: "MCC", class1: class1_train.mcc, class2: class2_test.mcc },
    { metric: "AUROC", class1: class1_train.auroc, class2: class2_test.auroc },
    { metric: "F1", class1: class1_train.f1, class2: class2_test.f1 },
  ];

  const bothPts = stability_scatter.filter((p) => p.sig === "both");
  const onePts = stability_scatter.filter((p) => p.sig === "one");
  const nsPts = stability_scatter.filter((p) => p.sig === "ns");

  return (
    <div className="space-y-5">
      {/* Side-by-side metrics */}
      <div className="bg-white rounded-lg border border-gray-100 p-5 shadow-sm">
        <h3 className="text-sm font-semibold text-gray-700 mb-3" style={{ fontFamily: "'VT323', monospace", textTransform: "uppercase", fontSize: "1rem" }}>
          Train on Class 1, Test on Class 2
        </h3>
        <p className="text-xs text-[#828282] mb-4" style={{ fontFamily: "'Roboto Condensed', sans-serif" }}>
          The probe was trained on one sequencing delivery (Class 1) and tested on a completely separate delivery (Class 2).
        </p>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200 text-left text-xs text-[#828282]">
                <th className="py-2 pr-4">Metric</th>
                <th className="py-2 pr-4 text-right">Class 1 (train)</th>
                <th className="py-2 pr-4 text-right">Class 2 (test)</th>
                <th className="py-2 text-right">Delta</th>
              </tr>
            </thead>
            <tbody>
              {comparisonData.map((row) => (
                <tr key={row.metric} className="border-b border-gray-50">
                  <td className="py-2.5 pr-4 font-medium">{row.metric}</td>
                  <td className="py-2.5 pr-4 text-right font-mono" style={{ color: "#0d8ba1" }}>{row.class1.toFixed(4)}</td>
                  <td className="py-2.5 pr-4 text-right font-mono" style={{ color: "#8a0038" }}>{row.class2.toFixed(4)}</td>
                  <td className="py-2.5 text-right font-mono text-[#828282]">{(row.class2 - row.class1).toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Dual ROC curves */}
      {cross_roc && (
        <div className="bg-white rounded-lg border border-gray-100 p-5 shadow-sm">
          <h3 className="text-sm font-semibold text-gray-700 mb-1" style={{ fontFamily: "'VT323', monospace", textTransform: "uppercase", fontSize: "1rem" }}>
            ROC Curves: Class 1 vs Class 2
          </h3>
          <p className="text-xs text-[#828282] mb-3" style={{ fontFamily: "'Roboto Condensed', sans-serif" }}>
            Nearly overlapping curves confirm the pathogenicity vector transfers across sequencing batches.
          </p>
          <ResponsiveContainer width="100%" height={400}>
            <ScatterChart margin={{ top: 10, right: 20, bottom: 30, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis type="number" dataKey="fpr" domain={[0, 1]} tick={{ fontSize: 11 }}
                label={{ value: "False Positive Rate", position: "bottom", offset: 15, style: { fontSize: 12 } }} />
              <YAxis type="number" dataKey="tpr" domain={[0, 1]} tick={{ fontSize: 11 }}
                label={{ value: "True Positive Rate", angle: -90, position: "insideLeft", offset: 10, style: { fontSize: 12 } }} />
              <Tooltip formatter={(v) => v.toFixed(3)} />
              <Scatter name="Class 1 (train)" data={cross_roc.class1} line={{ stroke: "#0d8ba1", strokeWidth: 2 }} fill="none" legendType="none" />
              <Scatter name="Class 2 (test)" data={cross_roc.class2} line={{ stroke: "#8a0038", strokeWidth: 2 }} fill="none" legendType="none" />
              <Scatter data={[{ fpr: 0, tpr: 0 }, { fpr: 1, tpr: 1 }]} line={{ stroke: "#d4d4d4", strokeWidth: 1, strokeDasharray: "5 5" }} fill="none" legendType="none" />
            </ScatterChart>
          </ResponsiveContainer>
          <div className="flex items-center justify-center gap-6 mt-2 text-xs text-[#828282]">
            <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-[#0d8ba1] inline-block" />Class 1 train (AUROC {class1_train.auroc.toFixed(3)})</span>
            <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-[#8a0038] inline-block" />Class 2 test (AUROC {class2_test.auroc.toFixed(3)})</span>
          </div>
        </div>
      )}

      {/* Feature stability scatter */}
      <div className="bg-white rounded-lg border border-gray-100 p-5 shadow-sm">
        <h3 className="text-sm font-semibold text-gray-700 mb-1" style={{ fontFamily: "'VT323', monospace", textTransform: "uppercase", fontSize: "1rem" }}>
          Feature Enrichment Stability
        </h3>
        <p className="text-xs text-[#828282] mb-3" style={{ fontFamily: "'Roboto Condensed', sans-serif" }}>
          Each point is one latent. Axes show log2(OR) in each class. Points on the diagonal have stable enrichment.
          Spearman rho = {feature_stability.spearman_all.toFixed(3)} (all), {feature_stability.spearman_significant.toFixed(3)} (significant).
        </p>
        <ResponsiveContainer width="100%" height={450}>
          <ScatterChart margin={{ top: 10, right: 20, bottom: 30, left: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis type="number" dataKey="x" domain={[-10, 10]} tick={{ fontSize: 11 }}
              label={{ value: "Class 1 log2(OR)", position: "bottom", offset: 15, style: { fontSize: 12 } }} />
            <YAxis type="number" dataKey="y" domain={[-10, 10]} tick={{ fontSize: 11 }}
              label={{ value: "Class 2 log2(OR)", angle: -90, position: "insideLeft", offset: 10, style: { fontSize: 12 } }} />
            <Tooltip content={<StabilityTooltip />} />
            <ReferenceLine slope={1} stroke="#000" strokeDasharray="5 5" strokeWidth={0.8} />
            <Scatter name="Not significant" data={nsPts} fill={COLORS.ns} fillOpacity={0.1} r={1} />
            <Scatter name="One class" data={onePts} fill={COLORS.one} fillOpacity={0.3} r={1.5} />
            <Scatter name="Both classes" data={bothPts} fill={COLORS.both} fillOpacity={0.4} r={2} />
          </ScatterChart>
        </ResponsiveContainer>
        <div className="flex items-center justify-center gap-6 mt-2 text-xs text-[#828282]">
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-[#8a0038] inline-block" />Significant in both ({bothPts.length.toLocaleString()})</span>
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-[#b88a00] inline-block" />Significant in one ({onePts.length.toLocaleString()})</span>
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-[#d4d4d4] inline-block" />Not significant ({nsPts.length.toLocaleString()})</span>
        </div>
      </div>

      {/* Key stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {[
          { label: "Sig in Class 1", value: feature_stability.n_sig_class1.toLocaleString(), color: "#0d8ba1" },
          { label: "Sig in Class 2", value: feature_stability.n_sig_class2.toLocaleString(), color: "#8a0038" },
          { label: "Sig in Both", value: feature_stability.n_sig_both.toLocaleString(), color: "#4e8c02" },
          { label: "Overlap Rate", value: ((feature_stability.n_sig_both / feature_stability.n_sig_either) * 100).toFixed(1) + "%", color: "#6b4fa0" },
        ].map((s) => (
          <div key={s.label} className="bg-white rounded-lg border border-gray-100 p-3 text-center shadow-sm">
            <div className="text-xl" style={{ fontFamily: "'VT323', monospace", color: s.color }}>{s.value}</div>
            <div className="text-xs text-[#828282]">{s.label}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function StabilityTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="bg-white border border-gray-200 rounded-md shadow-sm p-2 text-xs">
      <p>Class 1 log2(OR): {d.x.toFixed(3)}</p>
      <p>Class 2 log2(OR): {d.y.toFixed(3)}</p>
      <p className="capitalize">{d.sig === "both" ? "Significant in both" : d.sig === "one" ? "Significant in one" : "Not significant"}</p>
    </div>
  );
}
