export function Loading() {
  return (
    <div className="flex items-center justify-center h-64 text-[#828282]">
      <div className="text-center">
        <div className="w-8 h-8 border-2 border-[#0d8ba1] border-t-transparent rounded-full animate-spin mx-auto mb-3" />
        <p className="text-sm" style={{ fontFamily: "'Roboto Condensed', sans-serif" }}>Loading data...</p>
      </div>
    </div>
  );
}

export function ErrorState({ message }) {
  return (
    <div className="flex items-center justify-center h-64">
      <div className="text-center p-6 rounded-lg bg-red-50 border border-red-200 max-w-md">
        <p className="text-sm text-red-700 font-bold mb-1">Failed to load data</p>
        <p className="text-xs text-red-500" style={{ fontFamily: "'Roboto Condensed', sans-serif" }}>{message}</p>
        <p className="text-xs text-[#828282] mt-2">Data file may be missing or malformed. Check <code className="bg-red-100 px-1 rounded">viz/public/data/</code></p>
      </div>
    </div>
  );
}
