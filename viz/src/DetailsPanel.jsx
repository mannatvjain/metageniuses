import { Info, Tag, Layers, Zap, BarChart2 } from "lucide-react";

export default function DetailsPanel({ feature }) {
  return (
    <div className="w-72 flex-shrink-0 border-l border-gray-200 bg-white overflow-y-auto">
      <div className="p-4 space-y-5">
        {/* Feature info */}
        <div>
          <h3 className="text-xs font-semibold text-[#828282] uppercase tracking-wider mb-3 flex items-center gap-1.5" style={{ fontFamily: "'VT323', monospace", fontSize: "0.875rem" }}>
            <Info className="w-3 h-3" />
            Feature Info
          </h3>
          <dl className="space-y-2.5">
            <DetailRow icon={Tag} label="Category" value={feature.category} />
            <DetailRow icon={Layers} label="Layer" value={`Layer ${feature.layer}`} />
            <DetailRow icon={Zap} label="Freq Active" value={`${(feature.freqActive * 100).toFixed(1)}%`} />
            <DetailRow icon={BarChart2} label="Max Activation" value={feature.maxAct.toFixed(3)} />
          </dl>
        </div>

        {/* Top taxa */}
        <div className="pt-4 border-t border-gray-200">
          <h3 className="text-xs font-semibold text-[#828282] uppercase tracking-wider mb-3" style={{ fontFamily: "'VT323', monospace", fontSize: "0.875rem" }}>
            Top Associated Taxa
          </h3>
          <div className="space-y-1.5">
            {feature.topTaxa.map((taxon) => (
              <div
                key={taxon.name}
                className="flex items-center justify-between text-sm"
              >
                <span className="text-gray-700 truncate">{taxon.name}</span>
                <div className="flex items-center gap-2 flex-shrink-0">
                  <div className="w-16 h-1.5 bg-gray-100 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-[#0d8ba1] rounded-full"
                      style={{ width: `${taxon.score * 100}%` }}
                    />
                  </div>
                  <span className="text-xs text-[#828282] w-8 text-right">
                    {(taxon.score * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Activation pattern */}
        <div className="pt-4 border-t border-gray-200">
          <h3 className="text-xs font-semibold text-[#828282] uppercase tracking-wider mb-3" style={{ fontFamily: "'VT323', monospace", fontSize: "0.875rem" }}>
            Activation Pattern
          </h3>
          <span className="inline-block px-2 py-1 rounded text-xs font-bold bg-[#4e8c02]/10 text-[#4e8c02] border border-[#4e8c02]/20">
            {feature.activationPattern}
          </span>
          <p className="text-xs text-[#828282] mt-2">
            {patternDescription(feature.activationPattern)}
          </p>
        </div>
      </div>
    </div>
  );
}

function DetailRow({ icon: Icon, label, value }) {
  return (
    <div className="flex items-center justify-between text-sm">
      <span className="text-[#828282] flex items-center gap-1.5">
        <Icon className="w-3.5 h-3.5" />
        {label}
      </span>
      <span className="text-gray-800 font-bold">{value}</span>
    </div>
  );
}

function patternDescription(pattern) {
  const desc = {
    "Short Motif": "Activates on a short contiguous region (1-20 tokens).",
    "Domain": "Activates across a broad domain-level region.",
    "Point": "Single prominent activation site.",
    "Whole": "Activates across >80% of the sequence.",
    "Periodic": "Regular, repeating activation intervals.",
  };
  return desc[pattern] || "Unclassified activation pattern.";
}
