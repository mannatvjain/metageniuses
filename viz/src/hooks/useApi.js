import { useState, useEffect } from "react";

const BASE = import.meta.env.BASE_URL || "/";

function resolveUrl(url) {
  const map = {
    "/api/experiments/1": `${BASE}data/experiment1.json`,
    "/api/page/detectors": `${BASE}data/experiment1.json`,
    "/api/page/encoded-early/shared": `${BASE}data/page2_shared.json`,
    "/api/page/pathogenicity-vector": `${BASE}data/page3_pathogenicity_vector.json`,
    "/api/page/sae-health": `${BASE}data/page4_sae_health.json`,
  };
  if (map[url]) return map[url];

  let m = url.match(/^\/api\/page\/encoded-early\/volcano\/(\d+)$/);
  if (m) return `${BASE}data/page2_volcano_${m[1]}.json`;

  m = url.match(/^\/api\/experiments\/(\d+)$/);
  if (m) return `${BASE}data/experiment${m[1]}.json`;

  return url;
}

export default function useApi(url) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const staticUrl = resolveUrl(url);

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
