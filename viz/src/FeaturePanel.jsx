import SequenceStrip from "./SequenceStrip";

export default function FeaturePanel({ feature }) {
  return (
    <div className="flex-1 overflow-y-auto p-6 bg-gray-100">
      <div className="max-w-3xl mx-auto space-y-6">
        {/* Header */}
        <div>
          <h2 className="text-2xl">
            {feature.label}
            <span className="text-[#828282] ml-2 text-lg" style={{ fontFamily: "'Roboto Condensed', sans-serif", textTransform: "none" }}>
              Feature #{feature.id}
            </span>
          </h2>
          <p className="text-sm text-[#828282] mt-1">{feature.description}</p>
        </div>

        {/* Activation distribution */}
        <div className="rounded-lg bg-white shadow-sm border border-gray-100 p-4">
          <h3 className="text-sm font-semibold text-gray-700 mb-3" style={{ fontFamily: "'VT323', monospace", textTransform: "uppercase", fontSize: "1rem" }}>
            Activation Distribution
          </h3>
          <div className="h-32 flex items-end gap-1">
            {feature.histogram.map((val, i) => (
              <div
                key={i}
                className="flex-1 rounded-t"
                style={{
                  height: `${val * 100}%`,
                  backgroundColor: `rgba(13, 139, 161, ${0.3 + val * 0.7})`,
                }}
              />
            ))}
          </div>
          <div className="flex justify-between text-xs text-[#828282] mt-1">
            <span>0.0</span>
            <span>max activation</span>
          </div>
        </div>

        {/* Top activating sequences */}
        <div className="rounded-lg bg-white shadow-sm border border-gray-100 p-4">
          <h3 className="text-sm font-semibold text-gray-700 mb-3" style={{ fontFamily: "'VT323', monospace", textTransform: "uppercase", fontSize: "1rem" }}>
            Top Activating Sequences
          </h3>
          <div className="space-y-3">
            {feature.topSequences.map((seq, i) => (
              <SequenceStrip key={i} sequence={seq} rank={i + 1} />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
