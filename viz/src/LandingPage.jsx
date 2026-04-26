import { useState, useEffect, useRef } from "react";
import { Dna, Search, BarChart3, Shield, ArrowRight, FileText } from "lucide-react";

const ROTATING_WORDS = [
  { text: "Pathogenic Sequences", color: "#8a0038" },
  { text: "Gut Microbiomes", color: "#4e8c02" },
  { text: "Viral Signatures", color: "#0d8ba1" },
  { text: "Metagenomic Reads", color: "#828282" },
  { text: "Pandemic Signals", color: "#8a0038" },
  { text: "Interpretable Features", color: "#0d8ba1" },
];

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
  const [wordIndex, setWordIndex] = useState(0);
  const [visible, setVisible] = useState(true);

  useEffect(() => {
    const interval = setInterval(() => {
      setVisible(false);
      setTimeout(() => {
        setWordIndex((i) => (i + 1) % ROTATING_WORDS.length);
        setVisible(true);
      }, 300);
    }, 2500);
    return () => clearInterval(interval);
  }, []);

  const current = ROTATING_WORDS[wordIndex];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-white text-gray-900">
      {/* Hero */}
      <div className="max-w-4xl mx-auto px-8 pt-16 pb-12">
        <a
          href="/paper"
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-1.5 text-[#828282] hover:text-gray-600 transition-colors"
          style={{ fontFamily: "'VT323', monospace", textTransform: "uppercase", letterSpacing: "0.05em", fontSize: "1.4rem" }}
        >
          Read Our Paper <span>|</span>
        </a>
        <h1 className="tracking-tight mt-1 mb-4" style={{ fontSize: "4rem", lineHeight: 1.15, minHeight: "9.5rem" }}>
          MetaGenius Discovers{" "}
          <span
            className="transition-opacity duration-300"
            style={{ color: current.color, opacity: visible ? 1 : 0 }}
          >
            {current.text}
          </span>
        </h1>
        <p className="text-base text-[#828282] mb-8" style={{ fontFamily: "'Roboto Condensed', sans-serif" }}>
          Interpretable features from MetaGene-1 via sparse autoencoders.
          Understand what a metagenomic foundation model has learned about
          pathogens, microbiomes, and viral sequences.
        </p>
        <button
          onClick={onExplore}
          className="inline-flex items-center gap-2 px-5 py-2.5 rounded-2xl bg-slate-900 text-white text-sm font-medium hover:bg-slate-800 transition cursor-pointer tracking-wide"
          style={{ fontFamily: "'Roboto Condensed', sans-serif" }}
        >
          Explore Features
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>

      {/* Stats bar */}
      <div className="border-y border-gray-200 bg-white/70">
        <div className="max-w-5xl mx-auto px-6 py-4 grid grid-cols-2 md:grid-cols-4 gap-4">
          {STATS.map((s) => (
            <div key={s.label} className="text-center">
              <div className="text-2xl text-[#0d8ba1]" style={{ fontFamily: "'VT323', monospace" }}>{s.value}</div>
              <div className="text-xs text-[#828282] mt-0.5">{s.label}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Feature cards */}
      <div className="max-w-5xl mx-auto px-6 py-8 w-full">
        <div className="grid md:grid-cols-3 gap-5">
          {FEATURES.map((f) => (
            <div
              key={f.title}
              className="p-5 rounded-xl bg-white shadow-sm border border-gray-100 hover:shadow-md transition-shadow"
            >
              <f.icon className={`w-8 h-8 ${f.color} mb-3`} />
              <h3 className="text-xl mb-1.5">{f.title}</h3>
              <p className="text-xs text-[#828282]" style={{ fontFamily: "'Roboto Condensed', sans-serif" }}>{f.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Footer */}
      <div className="border-t border-gray-200 py-6 text-center text-xs text-[#828282]">
        MetaGeniuses — Apart Research AI x Bio Hackathon 2026
      </div>
    </div>
  );
}
