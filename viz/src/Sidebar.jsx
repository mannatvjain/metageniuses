import { useState } from "react";
import { Search, Layers, Zap } from "lucide-react";

export default function Sidebar({ features, selected, onSelect }) {
  const [query, setQuery] = useState("");

  const filtered = features.filter(
    (f) =>
      f.label.toLowerCase().includes(query.toLowerCase()) ||
      f.id.toString().includes(query)
  );

  return (
    <div className="w-64 flex-shrink-0 border-r border-gray-200 flex flex-col bg-white">
      {/* Search */}
      <div className="p-3 border-b border-gray-200">
        <div className="relative">
          <Search className="absolute left-2.5 top-2.5 w-3.5 h-3.5 text-[#828282]" />
          <input
            type="text"
            placeholder="Search features..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="w-full pl-8 pr-3 py-2 text-sm bg-gray-50 border border-gray-200 rounded-md text-gray-900 placeholder-[#828282] focus:outline-none focus:border-[#0d8ba1]"
          />
        </div>
      </div>

      {/* Feature list */}
      <div className="flex-1 overflow-y-auto">
        {filtered.map((f) => (
          <button
            key={f.id}
            onClick={() => onSelect(f)}
            className={`w-full text-left px-3 py-2.5 border-b border-gray-100 transition-colors cursor-pointer ${
              selected.id === f.id
                ? "bg-[#0d8ba1]/5 border-l-2 border-l-[#0d8ba1]"
                : "hover:bg-gray-50"
            }`}
          >
            <div className="flex items-center justify-between">
              <span className="text-sm font-bold text-gray-800">
                {f.label}
              </span>
              <span className="text-xs text-[#828282]">#{f.id}</span>
            </div>
            <div className="flex items-center gap-3 mt-1 text-xs text-[#828282]">
              <span className="flex items-center gap-1">
                <Layers className="w-3 h-3" />
                L{f.layer}
              </span>
              <span className="flex items-center gap-1">
                <Zap className="w-3 h-3" />
                {(f.freqActive * 100).toFixed(1)}%
              </span>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
