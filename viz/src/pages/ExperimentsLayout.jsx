import { NavLink, Outlet, Link } from "react-router-dom";
import { ArrowLeft } from "lucide-react";

const TABS = [
  { to: "/experiments/detectors", label: "Pathogen Detector Latents" },
  { to: "/experiments/encoded-early", label: "Encoded Early" },
  { to: "/experiments/pathogenicity-vector", label: "Pathogenicity Vector" },
  { to: "/experiments/sae-health", label: "SAE Health" },
];

export default function ExperimentsLayout() {
  return (
    <div className="min-h-screen bg-gray-100 text-gray-900">
      {/* Header */}
      <div className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-6 py-3 flex items-center gap-4">
          <Link
            to="/"
            className="inline-flex items-center gap-1 px-2 py-1 rounded-md border border-gray-200 bg-white hover:bg-gray-50 transition no-underline"
          >
            <ArrowLeft className="w-4 h-4 text-gray-500" />
            <span className="text-sm text-gray-600">Home</span>
          </Link>
          <h1 className="text-lg tracking-wide">
            Meta<span className="text-[#0d8ba1]">Geniuses</span>
            <span className="text-[#828282] ml-2 text-base" style={{ fontFamily: "'Roboto Condensed', sans-serif", textTransform: "none" }}>Results</span>
          </h1>
        </div>

        {/* Tabs */}
        <div className="max-w-7xl mx-auto px-6 flex gap-1 overflow-x-auto">
          {TABS.map((tab) => (
            <NavLink
              key={tab.to}
              to={tab.to}
              className={({ isActive }) =>
                `px-4 py-2.5 text-sm whitespace-nowrap border-b-2 transition-colors no-underline ${
                  isActive
                    ? "border-[#0d8ba1] text-[#0d8ba1] font-bold"
                    : "border-transparent text-[#828282] hover:text-gray-600"
                }`
              }
              style={{ fontFamily: "'Roboto Condensed', sans-serif" }}
            >
              {tab.label}
            </NavLink>
          ))}
        </div>
      </div>

      {/* Page content */}
      <div className="max-w-7xl mx-auto px-6 py-6">
        <Outlet />
      </div>
    </div>
  );
}
