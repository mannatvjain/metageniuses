export default function SequenceStrip({ sequence, rank }) {
  return (
    <div className="p-3 rounded-md bg-gray-50 border border-gray-200">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs text-[#828282]">#{rank}</span>
        <div className="flex items-center gap-2 text-xs">
          <span className="text-[#828282]">{sequence.source}</span>
          {sequence.classLabel && (
            <span className="px-1.5 py-0.5 rounded bg-[#0d8ba1]/10 text-[#0d8ba1] border border-[#0d8ba1]/20 font-bold">
              {sequence.classLabel}
            </span>
          )}
        </div>
      </div>
      {/* Sequence with activation coloring */}
      <div className="font-mono text-xs leading-relaxed tracking-wide overflow-x-auto" style={{ fontFamily: "'VT323', monospace" }}>
        {sequence.tokens.map((tok, i) => (
          <span
            key={i}
            className="inline-block"
            style={{
              backgroundColor: `rgba(13, 139, 161, ${tok.activation})`,
              borderRadius: "2px",
              padding: "1px 1px",
            }}
          >
            {tok.text}
          </span>
        ))}
      </div>
      <div className="flex items-center gap-4 mt-2 text-xs text-[#828282]">
        <span>Max act: {sequence.maxAct.toFixed(3)}</span>
        <span>{sequence.tokens.length} tokens</span>
      </div>
    </div>
  );
}
