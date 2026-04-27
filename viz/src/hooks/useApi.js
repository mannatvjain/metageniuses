import { useState, useEffect } from "react";

const BASE = import.meta.env.BASE_URL || "/";

const URL_MAP = {
  "/api/experiments/1": `${BASE}data/experiment1.json`,
  "/api/page/detectors": `${BASE}data/experiment1.json`,
  "/api/page/encoded-early": `${BASE}data/page2_encoded_early.json`,
  "/api/page/pathogenicity-vector": `${BASE}data/page3_pathogenicity_vector.json`,
  "/api/page/sae-health": `${BASE}data/page4_sae_health.json`,
};

export default function useApi(url) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const staticUrl = URL_MAP[url] || url.replace(/^\/api\/experiments\/(\d+)$/, `${BASE}data/experiment$1.json`);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    fetch(staticUrl)
      .then((res) => {
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
        return res.json();
      })
      .then((json) => {
        if (!cancelled) {
          setData(json);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err.message);
          setLoading(false);
        }
      });

    return () => { cancelled = true; };
  }, [staticUrl]);

  return { data, loading, error };
}
