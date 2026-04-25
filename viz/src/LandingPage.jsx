import { Dna, Search, BarChart3, Shield, ArrowRight } from "lucide-react";

const STATS = [
  { label: "Sequences Analyzed", value: "85,432" },
  { label: "SAE Features Learned", value: "4,096" },
  { label: "Layers Extracted", value: "4" },
  { label: "Pathogen Classes", value: "7" },
];

const FEATURES = [
  {
    icon: Search,
    title: "Feature Explorer",
    desc: "Browse sparse autoencoder features and see which sequences activate them.",
    color: "text-[#0d8ba1]",
  },
  {
    icon: BarChart3,
    title: "Activation Heatmaps",
    desc: "Visualize feature activations across sequences with per-token resolution.",
    color: "text-[#4e8c02]",
  },
  {
    icon: Shield,
    title: "Pathogen Detection",
    desc: "Map learned features to known pathogen classes for interpretable surveillance.",
    color: "text-[#8a0038]",
  },
];

export default function LandingPage({ onExplore }) {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-white text-gray-900">
      {/* Hero */}
      <div className="max-w-5xl mx-auto px-6 pt-24 pb-16 text-center">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-gray-200 text-[#828282] text-sm mb-6">
          <Dna className="w-4 h-4 text-[#0d8ba1]" />
          Apart Research AI x Bio Hackathon
        </div>
        <h1 className="text-6xl tracking-tight mb-4">
          Meta<span className="text-[#0d8ba1]">Geniuses</span>
        </h1>
        <p className="text-lg text-[#828282] max-w-2xl mx-auto mb-8" style={{ fontFamily: "'Roboto Condensed', sans-serif" }}>
          Interpretable features from MetaGene-1 via sparse autoencoders.
          Understand what a metagenomic foundation model has learned about
          pathogens, microbiomes, and viral sequences.
        </p>
        <button
          onClick={onExplore}
          className="inline-flex items-center gap-2 px-6 py-3 rounded-2xl bg-slate-900 text-white font-medium hover:bg-slate-800 transition cursor-pointer tracking-wide"
          style={{ fontFamily: "'Roboto Condensed', sans-serif" }}
        >
          Explore Features
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>

      {/* Stats bar */}
      <div className="border-y border-gray-200 bg-white/70">
        <div className="max-w-5xl mx-auto px-6 py-6 grid grid-cols-2 md:grid-cols-4 gap-6">
          {STATS.map((s) => (
            <div key={s.label} className="text-center">
              <div className="text-3xl text-[#0d8ba1]" style={{ fontFamily: "'VT323', monospace" }}>{s.value}</div>
              <div className="text-sm text-[#828282] mt-1">{s.label}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Feature cards */}
      <div className="max-w-5xl mx-auto px-6 py-16">
        <div className="grid md:grid-cols-3 gap-6">
          {FEATURES.map((f) => (
            <div
              key={f.title}
              className="p-6 rounded-xl bg-white shadow-sm border border-gray-100 hover:shadow-md transition-shadow"
            >
              <f.icon className={`w-8 h-8 ${f.color} mb-4`} />
              <h3 className="text-xl mb-2">{f.title}</h3>
              <p className="text-sm text-[#828282]" style={{ fontFamily: "'Roboto Condensed', sans-serif" }}>{f.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Footer */}
      <div className="border-t border-gray-200 py-8 text-center text-sm text-[#828282]">
        MetaGeniuses — Apart Research AI x Bio Hackathon 2026
      </div>
    </div>
  );
}
