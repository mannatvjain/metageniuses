import { useState } from "react";
import LandingPage from "./LandingPage";
import ExplorerView from "./ExplorerView";

export default function App() {
  const [view, setView] = useState("landing"); // "landing" | "explorer"

  if (view === "landing") {
    return <LandingPage onExplore={() => setView("explorer")} />;
  }

  return <ExplorerView onBack={() => setView("landing")} />;
}
