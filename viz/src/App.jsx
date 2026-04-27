import { lazy, Suspense } from "react";
import { HashRouter, Routes, Route } from "react-router-dom";
import { Loading } from "./components/LoadingState";

const LandingPage = lazy(() => import("./LandingPage"));
const ExperimentsLayout = lazy(() => import("./pages/ExperimentsLayout"));
const Experiment1 = lazy(() => import("./pages/Experiment1"));
const EncodedEarly = lazy(() => import("./pages/EncodedEarly"));
const PathogenicityVector = lazy(() => import("./pages/PathogenicityVector"));
const SAEHealth = lazy(() => import("./pages/SAEHealth"));

export default function App() {
  return (
    <HashRouter>
      <Suspense fallback={<Loading />}>
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/experiments" element={<ExperimentsLayout />}>
            <Route index element={<Experiment1 />} />
            <Route path="detectors" element={<Experiment1 />} />
            <Route path="encoded-early" element={<EncodedEarly />} />
            <Route path="pathogenicity-vector" element={<PathogenicityVector />} />
            <Route path="sae-health" element={<SAEHealth />} />
          </Route>
        </Routes>
      </Suspense>
    </HashRouter>
  );
}
