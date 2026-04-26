import { lazy, Suspense } from "react";
import { HashRouter, Routes, Route } from "react-router-dom";
import { Loading } from "./components/LoadingState";

const LandingPage = lazy(() => import("./LandingPage"));
const ExplorerView = lazy(() => import("./ExplorerView"));
const ExperimentsLayout = lazy(() => import("./pages/ExperimentsLayout"));
const Experiment1 = lazy(() => import("./pages/Experiment1"));
const Experiment2 = lazy(() => import("./pages/Experiment2"));
const Experiment3 = lazy(() => import("./pages/Experiment3"));
const Experiment4 = lazy(() => import("./pages/Experiment4"));
const Experiment5 = lazy(() => import("./pages/Experiment5"));

export default function App() {
  return (
    <HashRouter>
      <Suspense fallback={<Loading />}>
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/explorer" element={<ExplorerView />} />
          <Route path="/experiments" element={<ExperimentsLayout />}>
            <Route index element={<Experiment1 />} />
            <Route path="1" element={<Experiment1 />} />
            <Route path="2" element={<Experiment2 />} />
            <Route path="3" element={<Experiment3 />} />
            <Route path="4" element={<Experiment4 />} />
            <Route path="5" element={<Experiment5 />} />
          </Route>
        </Routes>
      </Suspense>
    </HashRouter>
  );
}
