import { useState } from "react";
import { ArrowLeft } from "lucide-react";
import Sidebar from "./Sidebar";
import FeaturePanel from "./FeaturePanel";
import DetailsPanel from "./DetailsPanel";
import { DUMMY_FEATURES } from "./data/dummyFeatures";

export default function ExplorerView({ onBack }) {
  const [selectedFeature, setSelectedFeature] = useState(DUMMY_FEATURES[0]);

  return (
    <div className="h-screen flex flex-col bg-gray-100 text-gray-900">
      {/* Top bar */}
      <div className="flex items-center gap-3 px-4 py-3 border-b border-gray-200 bg-white">
        <button
          onClick={onBack}
          className="inline-flex items-center gap-1 px-2 py-1 rounded-md border border-gray-200 bg-white hover:bg-gray-50 transition cursor-pointer"
        >
          <ArrowLeft className="w-4 h-4 text-gray-500" />
          <span className="text-sm text-gray-600">Back</span>
        </button>
        <h1 className="text-lg tracking-wide">
          Meta<span className="text-[#0d8ba1]">Geniuses</span>
          <span className="text-[#828282] ml-2 text-base" style={{ fontFamily: "'Roboto Condensed', sans-serif", textTransform: "none" }}>Feature Explorer</span>
        </h1>
      </div>

      {/* Main layout: sidebar | feature viz | detail panel */}
      <div className="flex flex-1 overflow-hidden">
        <Sidebar
          features={DUMMY_FEATURES}
          selected={selectedFeature}
          onSelect={setSelectedFeature}
        />
        <FeaturePanel feature={selectedFeature} />
        <DetailsPanel feature={selectedFeature} />
      </div>
    </div>
  );
}
