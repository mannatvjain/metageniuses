"""
Experiment 1: Organism-Specific Pathogen Detectors
See experiment_plans/01_organism_detectors.md for full spec.

Finds SAE latents that fire specifically on pathogen sequences,
retrieves top-activating sequences, BLASTs them against NCBI,
and labels each latent with the organism it detects.

Usage:
    # Parts A+B only (local, fast):
    python experiments/organism_detectors.py --parts AB

    # Test BLAST with 3 latents x 3 sequences:
    python experiments/organism_detectors.py --parts C --blast-test

    # Full BLAST (50 latents x 10 sequences):
    python experiments/organism_detectors.py --parts C

    # Organism labeling + figures:
    python experiments/organism_detectors.py --parts DE

    # Everything:
    python experiments/organism_detectors.py --parts ABCDE
"""

import argparse
import io
import json
import time
import sys
import urllib.parse
import urllib.request
import urllib.error
import http.client
import zipfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

# ---------- Config ----------
# Defaults for layer 32; overridden by --layer flag in main()
DATA_DIR = Path("data/sae_model")
LABEL_FILE = Path("data/human_virus_class1_labeled.jsonl")
OUT_DIR = Path("results/organism_detectors")

LAYER_DATA_DIRS = {
    8: Path("data/sae_layer8"),
    16: Path("data/sae_layer16"),
    24: Path("data/sae_layer24"),
    32: Path("data/sae_model"),
}

BLAST_URL = "https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi"
BLAST_EMAIL = "mannat.v.jain@columbia.edu"
BLAST_TOOL = "metageniuses"

TOP_LATENTS_CAP = 50
TOP_SEQUENCES_PER_LATENT = 10
F1_THRESHOLD = 0.7
FDR_ALPHA = 0.01

BLAST_SUBMIT_DELAY = 0.5     # seconds between submissions
BLAST_POLL_INTERVAL = 15     # seconds between polls
BLAST_TIMEOUT = 600           # 10 min per job
BLAST_MAX_RETRIES = 3


# ============================================================
#  DATA LOADING
# ============================================================

def load_data():
    """Load features, labels, and sequence lookup."""
    print("Loading features...")
    features = np.load(DATA_DIR / "features.npy")
    with open(DATA_DIR / "sequence_ids.json") as f:
        sequence_ids = json.load(f)

    # Build label lookup and sequence lookup
    label_lookup = {}
    seq_lookup = {}
    with open(LABEL_FILE) as f:
        for line in f:
            row = json.loads(line)
            label_lookup[row["sequence_id"]] = int(row["source"])
            seq_lookup[row["sequence_id"]] = row["sequence"]

    labels = np.array([label_lookup[sid] for sid in sequence_ids])
    pathogen = labels == 1

    print(f"Features: {features.shape}, Labels: {labels.shape}")
    print(f"Class balance: pathogen={pathogen.sum()}, non-pathogen={(~pathogen).sum()}")
    return features, sequence_ids, labels, pathogen, seq_lookup


# ============================================================
#  PART A: ENRICHMENT SCAN
# ============================================================

def _fast_f1(col, pathogen_int, n_pos):
    """Compute best F1 across 19 thresholds without sklearn overhead."""
    max_val = col.max()
    if max_val == 0:
        return 0.0
    best_f1 = 0.0
    n = len(col)
    for frac in np.arange(0.05, 1.0, 0.05):
        threshold = frac * max_val
        pred = col >= threshold
        tp = np.sum(pred & (pathogen_int == 1))
        fp = np.sum(pred & (pathogen_int == 0))
        fn = n_pos - tp
        if tp == 0:
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > best_f1:
            best_f1 = f1
    return best_f1


def run_part_a(features, pathogen):
    """Compute enrichment metrics for all 32,768 latents.

    Optimized: vectorized activation counts/means, manual F1, batched Wilcoxon.
    """
    from scipy.stats import fisher_exact, mannwhitneyu
    from statsmodels.stats.multitest import multipletests

    n_latents = features.shape[1]
    n_samples = features.shape[0]
    print(f"\n{'='*60}")
    print(f"PART A: Enrichment scan across {n_latents} latents")
    print(f"{'='*60}")

    path_idx = np.where(pathogen)[0]
    nonpath_idx = np.where(~pathogen)[0]
    n_pos = len(path_idx)
    n_neg = len(nonpath_idx)
    pathogen_int = pathogen.astype(int)

    # Vectorized: activation counts and means per class
    print("  Computing vectorized activation counts and means...")
    active_matrix = features > 0  # (20000, 32768) bool
    act_count_p = active_matrix[path_idx].sum(axis=0).astype(int)
    act_count_np = active_matrix[nonpath_idx].sum(axis=0).astype(int)
    mean_act_p = features[path_idx].mean(axis=0)
    mean_act_np = features[nonpath_idx].mean(axis=0)

    # Vectorized: log2FC
    eps = 1e-10
    log2fcs = np.log2((mean_act_p + eps) / (mean_act_np + eps))

    # Fisher's exact test + Wilcoxon + F1 sweep (loop, but optimized)
    fisher_ors = np.zeros(n_latents)
    fisher_ps = np.zeros(n_latents)
    wilcox_ps = np.ones(n_latents)
    best_f1s = np.zeros(n_latents)

    # Pre-extract pathogen/nonpathogen feature slices
    features_p = features[path_idx]    # (n_pos, n_latents)
    features_np = features[nonpath_idx]  # (n_neg, n_latents)

    start = time.time()
    for i in range(n_latents):
        if i % 2000 == 0:
            elapsed = time.time() - start
            rate = i / max(elapsed, 1)
            eta = (n_latents - i) / max(rate, 1)
            print(f"  Latent {i:>6}/{n_latents}  ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

        a = int(act_count_p[i])
        b = int(act_count_np[i])
        c = n_pos - a
        d = n_neg - b

        # Fisher's exact test
        try:
            odds_ratio, p_value = fisher_exact([[a, b], [c, d]])
        except Exception:
            odds_ratio, p_value = 1.0, 1.0
        fisher_ors[i] = odds_ratio
        fisher_ps[i] = p_value

        # Wilcoxon (skip if both groups are all-zero)
        col_p = features_p[:, i]
        col_np = features_np[:, i]
        if mean_act_p[i] == 0 and mean_act_np[i] == 0:
            wilcox_ps[i] = 1.0
        else:
            try:
                _, wp = mannwhitneyu(col_p, col_np, alternative='two-sided')
                wilcox_ps[i] = wp
            except Exception:
                wilcox_ps[i] = 1.0

        # F1 sweep (manual, no sklearn overhead)
        best_f1s[i] = _fast_f1(features[:, i], pathogen_int, n_pos)

    elapsed = time.time() - start
    print(f"  Enrichment scan complete in {elapsed:.1f}s")

    # FDR correction
    print("  Applying FDR correction...")
    _, fisher_fdrs, _, _ = multipletests(fisher_ps, alpha=FDR_ALPHA, method='fdr_bh')
    _, wilcox_fdrs, _, _ = multipletests(wilcox_ps, alpha=FDR_ALPHA, method='fdr_bh')

    # Classification
    is_path_enriched = (fisher_fdrs < FDR_ALPHA) & (fisher_ors > 1)
    is_nonpath_enriched = (fisher_fdrs < FDR_ALPHA) & (fisher_ors < 1)
    is_path_specific = best_f1s > F1_THRESHOLD

    print(f"\n  SUMMARY:")
    print(f"    Pathogen-enriched (Fisher FDR<0.01, OR>1):   {is_path_enriched.sum()}")
    print(f"    Non-pathogen-enriched (Fisher FDR<0.01, OR<1): {is_nonpath_enriched.sum()}")
    print(f"    Pathogen-specific (F1 > {F1_THRESHOLD}):          {is_path_specific.sum()}")
    print(f"    Wilcoxon significant (FDR<0.01):             {(wilcox_fdrs < FDR_ALPHA).sum()}")

    # Save CSV
    import csv
    csv_path = OUT_DIR / "enrichment_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "latent_id", "fisher_or", "fisher_p", "fisher_fdr",
            "log2fc", "wilcoxon_p", "wilcoxon_fdr", "best_f1",
            "is_pathogen_enriched", "is_nonpathogen_enriched", "is_pathogen_specific",
            "activation_count_pathogen", "activation_count_nonpathogen",
            "mean_activation_pathogen", "mean_activation_nonpathogen",
        ])
        for i in range(n_latents):
            writer.writerow([
                i, fisher_ors[i], fisher_ps[i], fisher_fdrs[i],
                log2fcs[i], wilcox_ps[i], wilcox_fdrs[i], best_f1s[i],
                bool(is_path_enriched[i]), bool(is_nonpath_enriched[i]),
                bool(is_path_specific[i]),
                int(act_count_p[i]), int(act_count_np[i]),
                mean_act_p[i], mean_act_np[i],
            ])
    print(f"  Saved: {csv_path}")

    return {
        "fisher_ors": fisher_ors,
        "fisher_fdrs": fisher_fdrs,
        "log2fcs": log2fcs,
        "wilcox_fdrs": wilcox_fdrs,
        "best_f1s": best_f1s,
        "is_path_enriched": is_path_enriched,
        "is_nonpath_enriched": is_nonpath_enriched,
        "is_path_specific": is_path_specific,
        "act_count_p": act_count_p,
        "act_count_np": act_count_np,
        "mean_act_p": mean_act_p,
        "mean_act_np": mean_act_np,
    }


def load_enrichment_from_csv():
    """Load previously computed enrichment results from CSV."""
    import csv
    csv_path = OUT_DIR / "enrichment_results.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run Part A first.")
        sys.exit(1)

    print(f"Loading enrichment results from {csv_path}...")
    n = 32768
    result = {
        "fisher_ors": np.zeros(n),
        "fisher_fdrs": np.zeros(n),
        "log2fcs": np.zeros(n),
        "wilcox_fdrs": np.zeros(n),
        "best_f1s": np.zeros(n),
        "is_path_enriched": np.zeros(n, dtype=bool),
        "is_nonpath_enriched": np.zeros(n, dtype=bool),
        "is_path_specific": np.zeros(n, dtype=bool),
        "act_count_p": np.zeros(n, dtype=int),
        "act_count_np": np.zeros(n, dtype=int),
        "mean_act_p": np.zeros(n),
        "mean_act_np": np.zeros(n),
    }

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            i = int(row["latent_id"])
            result["fisher_ors"][i] = float(row["fisher_or"])
            result["fisher_fdrs"][i] = float(row["fisher_fdr"])
            result["log2fcs"][i] = float(row["log2fc"])
            result["wilcox_fdrs"][i] = float(row["wilcoxon_fdr"])
            result["best_f1s"][i] = float(row["best_f1"])
            result["is_path_enriched"][i] = row["is_pathogen_enriched"] == "True"
            result["is_nonpath_enriched"][i] = row["is_nonpathogen_enriched"] == "True"
            result["is_path_specific"][i] = row["is_pathogen_specific"] == "True"
            result["act_count_p"][i] = int(row["activation_count_pathogen"])
            result["act_count_np"][i] = int(row["activation_count_nonpathogen"])
            result["mean_act_p"][i] = float(row["mean_activation_pathogen"])
            result["mean_act_np"][i] = float(row["mean_activation_nonpathogen"])

    return result


# ============================================================
#  PART B: SEQUENCE RETRIEVAL
# ============================================================

def run_part_b(features, sequence_ids, pathogen, seq_lookup, enrichment,
               max_latents=TOP_LATENTS_CAP, max_seqs=TOP_SEQUENCES_PER_LATENT):
    """Select top latents and retrieve top-activating pathogen sequences."""
    print(f"\n{'='*60}")
    print(f"PART B: Sequence retrieval (top {max_latents} latents x {max_seqs} sequences)")
    print(f"{'='*60}")

    # Select latents: first pathogen-specific, then pathogen-enriched by OR
    specific_ids = np.where(enrichment["is_path_specific"])[0]
    enriched_ids = np.where(
        enrichment["is_path_enriched"] & ~enrichment["is_path_specific"]
    )[0]

    # Sort enriched by Fisher OR descending
    enriched_ids = enriched_ids[np.argsort(enrichment["fisher_ors"][enriched_ids])[::-1]]

    # Combine: specific first, then enriched, cap at max_latents
    selected = list(specific_ids)
    for lid in enriched_ids:
        if len(selected) >= max_latents:
            break
        selected.append(lid)
    selected = selected[:max_latents]

    print(f"  Selected {len(selected)} latents ({len(specific_ids)} specific + "
          f"{len(selected) - len(specific_ids)} enriched)")

    # Pathogen sequence indices
    path_mask = pathogen
    path_indices = np.where(path_mask)[0]

    result = {}
    for lid in selected:
        col = features[:, lid]
        # Sort pathogen sequences by activation, descending
        path_activations = col[path_indices]
        top_local = np.argsort(path_activations)[::-1][:max_seqs]
        top_global = path_indices[top_local]

        seqs = []
        for gi in top_global:
            sid = sequence_ids[gi]
            seqs.append({
                "sequence_id": sid,
                "activation": float(col[gi]),
                "sequence": seq_lookup.get(sid, ""),
            })

        result[str(lid)] = {
            "fisher_or": float(enrichment["fisher_ors"][lid]),
            "fisher_fdr": float(enrichment["fisher_fdrs"][lid]),
            "log2fc": float(enrichment["log2fcs"][lid]),
            "best_f1": float(enrichment["best_f1s"][lid]),
            "top_sequences": seqs,
        }

    out_path = OUT_DIR / "top_sequences_per_latent.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out_path}")
    print(f"  Total sequences to BLAST: {sum(len(v['top_sequences']) for v in result.values())}")

    return result


# ============================================================
#  PART C: BLAST
# ============================================================

def _blast_request(params, method="POST", retries=BLAST_MAX_RETRIES, raw=False):
    """Make an HTTP request to NCBI BLAST API with retries.
    If raw=True, return bytes instead of decoded text.
    """
    params["TOOL"] = BLAST_TOOL
    params["EMAIL"] = BLAST_EMAIL

    for attempt in range(retries):
        try:
            if method == "POST":
                data = urllib.parse.urlencode(params).encode("utf-8")
                req = urllib.request.Request(BLAST_URL, data=data)
            else:
                url = BLAST_URL + "?" + urllib.parse.urlencode(params)
                req = urllib.request.Request(url)

            with urllib.request.urlopen(req, timeout=60) as resp:
                body = resp.read()
                return body if raw else body.decode("utf-8", errors="replace")

        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503):
                wait = (2 ** attempt) * 5
                print(f"    HTTP {e.code}, backing off {wait}s (attempt {attempt+1}/{retries})")
                time.sleep(wait)
            else:
                raise
        except (urllib.error.URLError, TimeoutError, http.client.IncompleteRead) as e:
            wait = (2 ** attempt) * 5
            print(f"    Network error: {e}, backing off {wait}s (attempt {attempt+1}/{retries})")
            time.sleep(wait)

    raise RuntimeError(f"BLAST request failed after {retries} retries")


def _submit_blast(sequence):
    """Submit a BLAST search and return the RID."""
    params = {
        "CMD": "Put",
        "PROGRAM": "blastn",
        "DATABASE": "nt",
        "QUERY": sequence,
    }
    resp = _blast_request(params, method="POST")

    # Parse RID from response
    rid = None
    for line in resp.split("\n"):
        if "RID = " in line:
            rid = line.split("RID = ")[1].strip()
            break
    if not rid:
        raise RuntimeError(f"Could not parse RID from BLAST response:\n{resp[:500]}")
    return rid


def _poll_blast(rid, timeout=BLAST_TIMEOUT):
    """Poll for BLAST results. Returns parsed JSON or status string."""
    start = time.time()
    while time.time() - start < timeout:
        time.sleep(BLAST_POLL_INTERVAL)

        # Check status first
        params_check = {"CMD": "Get", "RID": rid, "FORMAT_OBJECT": "SearchInfo"}
        try:
            resp_check = _blast_request(params_check, method="GET")
        except RuntimeError:
            continue

        if "Status=WAITING" in resp_check:
            elapsed = int(time.time() - start)
            print(f"      RID {rid}: WAITING ({elapsed}s)")
            continue
        elif "Status=FAILED" in resp_check:
            return {"status": "failed"}
        elif "Status=UNKNOWN" in resp_check:
            return {"status": "unknown"}
        elif "Status=READY" in resp_check:
            # Fetch actual results (NCBI returns a ZIP for JSON2)
            params_get = {
                "CMD": "Get",
                "FORMAT_TYPE": "JSON2",
                "RID": rid,
            }
            try:
                result_bytes = _blast_request(params_get, method="GET", raw=True)
                # Response is a ZIP containing JSON files
                if result_bytes[:2] == b"PK":
                    with zipfile.ZipFile(io.BytesIO(result_bytes)) as zf:
                        # Find the actual results JSON (not the index)
                        json_files = [n for n in zf.namelist() if n.endswith("_1.json")]
                        if not json_files:
                            json_files = [n for n in zf.namelist() if n.endswith(".json")]
                        with zf.open(json_files[-1]) as jf:
                            return json.loads(jf.read())
                else:
                    return json.loads(result_bytes)
            except (json.JSONDecodeError, RuntimeError, zipfile.BadZipFile, KeyError) as e:
                print(f"      RID {rid}: Failed to parse results: {e}")
                return {"status": "parse_error"}

    return {"status": "timeout"}


def _parse_blast_hit(result_json):
    """Extract top hit info from BLAST JSON2 response."""
    try:
        bo2 = result_json["BlastOutput2"]
        # Handle both list format and dict format
        report = bo2[0]["report"] if isinstance(bo2, list) else bo2["report"]
        search = report["results"]["search"]
        hits = search.get("hits", [])
        if not hits:
            return {"status": "no_hit"}

        top_hit = hits[0]
        description = top_hit["description"][0]
        hsps = top_hit["hsps"][0]

        organism = description.get("sciname", description.get("title", "unknown"))
        accession = description.get("accession", "")

        # Query coverage: align_len / query_len
        query_len = search.get("query_len", 1)
        align_len = hsps.get("align_len", 0)
        query_coverage = round(100 * align_len / query_len) if query_len > 0 else 0

        return {
            "status": "hit",
            "top_hit": {
                "organism": organism,
                "accession": accession,
                "description": description.get("title", ""),
                "percent_identity": round(100 * hsps.get("identity", 0) / hsps.get("align_len", 1), 1) if hsps.get("align_len", 0) > 0 else 0,
                "e_value": hsps.get("evalue", None),
                "query_coverage": query_coverage,
                "bit_score": hsps.get("bit_score", 0),
            },
        }
    except (KeyError, IndexError, TypeError) as e:
        return {"status": "parse_error", "error": str(e)}


def run_part_c(top_sequences, blast_test=False):
    """BLAST top-activating sequences against NCBI nt database.

    Parallelized: submit ALL sequences upfront, then poll in round-robin.
    """
    # Determine scope
    if blast_test:
        max_latents = 3
        max_seqs = 3
        print(f"\n{'='*60}")
        print(f"PART C: BLAST TEST RUN ({max_latents} latents x {max_seqs} sequences)")
        print(f"{'='*60}")
    else:
        max_latents = TOP_LATENTS_CAP
        max_seqs = TOP_SEQUENCES_PER_LATENT
        print(f"\n{'='*60}")
        print(f"PART C: BLAST FULL RUN ({max_latents} latents x {max_seqs} sequences)")
        print(f"{'='*60}")

    # Load checkpoint if exists
    partial_path = OUT_DIR / "blast_results_partial.json"
    if partial_path.exists():
        with open(partial_path) as f:
            blast_results = json.load(f)
        print(f"  Loaded checkpoint: {len(blast_results)} latents already done")
    else:
        blast_results = {}

    # Select latents, skip already-done
    latent_ids = list(top_sequences.keys())[:max_latents]
    remaining = [lid for lid in latent_ids if lid not in blast_results]
    print(f"  Total latents: {len(latent_ids)}, already done: {len(blast_results)}, remaining: {len(remaining)}")

    if not remaining:
        print("  All latents already done!")
    else:
        # Phase 1: Submit ALL sequences for all remaining latents upfront
        print(f"\n  --- Phase 1: Submitting all sequences ---")
        # jobs: list of (latent_id, seq_idx, seq_info, rid, short_query)
        jobs = []
        failed = []  # (latent_id, seq_info, error)
        for latent_id in remaining:
            seqs = top_sequences[latent_id]["top_sequences"][:max_seqs]
            for seq_idx, seq_info in enumerate(seqs):
                sequence = seq_info["sequence"]
                short_query = len(sequence) < 50
                try:
                    rid = _submit_blast(sequence)
                    jobs.append((latent_id, seq_idx, seq_info, rid, short_query))
                    time.sleep(BLAST_SUBMIT_DELAY)
                except RuntimeError as e:
                    print(f"    SUBMIT FAILED: L{latent_id} seq {seq_idx+1}: {e}")
                    failed.append((latent_id, seq_info, str(e)))

            print(f"    Submitted L{latent_id} ({len(seqs)} seqs)")

        total_submitted = len(jobs)
        print(f"  Total submitted: {total_submitted} sequences ({len(failed)} failures)")

        # Phase 2: Poll all RIDs in round-robin until all complete
        print(f"\n  --- Phase 2: Polling all {total_submitted} jobs (round-robin) ---")
        # Track results per job
        results_by_job = {}  # index -> parsed result
        pending = set(range(len(jobs)))
        start_time = time.time()

        # Wait a bit for NCBI to process before first poll
        print(f"    Waiting 30s before first poll...")
        time.sleep(30)

        poll_round = 0
        while pending:
            poll_round += 1
            newly_done = []
            elapsed = int(time.time() - start_time)
            print(f"    Poll round {poll_round}: {len(pending)} pending, {elapsed}s elapsed")

            for idx in sorted(pending):
                latent_id, seq_idx, seq_info, rid, short_query = jobs[idx]

                # Quick status check
                params_check = {"CMD": "Get", "RID": rid, "FORMAT_OBJECT": "SearchInfo"}
                try:
                    resp_check = _blast_request(params_check, method="GET")
                except RuntimeError:
                    continue

                if "Status=WAITING" in resp_check:
                    continue
                elif "Status=READY" in resp_check:
                    # Fetch results
                    params_get = {"CMD": "Get", "FORMAT_TYPE": "JSON2", "RID": rid}
                    try:
                        result_bytes = _blast_request(params_get, method="GET", raw=True)
                        if result_bytes[:2] == b"PK":
                            with zipfile.ZipFile(io.BytesIO(result_bytes)) as zf:
                                json_files = [n for n in zf.namelist() if n.endswith("_1.json")]
                                if not json_files:
                                    json_files = [n for n in zf.namelist() if n.endswith(".json")]
                                with zf.open(json_files[-1]) as jf:
                                    result_json = json.loads(jf.read())
                        else:
                            result_json = json.loads(result_bytes)

                        parsed = _parse_blast_hit(result_json)
                        parsed["sequence_id"] = seq_info["sequence_id"]
                        if short_query:
                            parsed["short_query"] = True
                        results_by_job[idx] = parsed

                        if parsed["status"] == "hit":
                            org = parsed["top_hit"]["organism"]
                            ident = parsed["top_hit"]["percent_identity"]
                            print(f"      L{latent_id} seq {seq_idx+1}: {org} ({ident}%)")
                        else:
                            print(f"      L{latent_id} seq {seq_idx+1}: {parsed['status']}")
                    except (json.JSONDecodeError, RuntimeError, zipfile.BadZipFile, KeyError) as e:
                        results_by_job[idx] = {"sequence_id": seq_info["sequence_id"], "status": "parse_error"}
                        print(f"      L{latent_id} seq {seq_idx+1}: parse_error ({e})")

                    newly_done.append(idx)
                elif "Status=FAILED" in resp_check or "Status=UNKNOWN" in resp_check:
                    status = "failed" if "FAILED" in resp_check else "unknown"
                    results_by_job[idx] = {"sequence_id": seq_info["sequence_id"], "status": status}
                    newly_done.append(idx)
                    print(f"      L{latent_id} seq {seq_idx+1}: {status}")

                # Small delay between status checks to be polite
                time.sleep(0.3)

            pending -= set(newly_done)

            # Check for overall timeout (10 min per remaining job is excessive)
            if time.time() - start_time > BLAST_TIMEOUT * 2:
                print(f"    Overall timeout reached. Marking {len(pending)} jobs as timeout.")
                for idx in pending:
                    _, _, seq_info, _, _ = jobs[idx]
                    results_by_job[idx] = {"sequence_id": seq_info["sequence_id"], "status": "timeout"}
                pending.clear()
                break

            if pending:
                print(f"    {len(pending)} still waiting, sleeping {BLAST_POLL_INTERVAL}s...")
                time.sleep(BLAST_POLL_INTERVAL)

        # Phase 3: Assemble results per latent and checkpoint
        print(f"\n  --- Phase 3: Assembling results ---")
        for latent_id in remaining:
            hits = []
            # Add failed submissions
            for flid, seq_info, error in failed:
                if flid == latent_id:
                    hits.append({"sequence_id": seq_info["sequence_id"], "status": "submit_failed", "error": error})

            # Add polled results
            for idx, (lid, _, _, _, _) in enumerate(jobs):
                if lid == latent_id and idx in results_by_job:
                    hits.append(results_by_job[idx])

            blast_results[latent_id] = {
                "sequences_submitted": len(top_sequences[latent_id]["top_sequences"][:max_seqs]),
                "sequences_with_hits": sum(1 for h in hits if h.get("status") == "hit"),
                "hits": hits,
            }

        # Save checkpoint
        with open(partial_path, "w") as f:
            json.dump(blast_results, f, indent=2)
        print(f"  Checkpoint saved ({len(blast_results)} latents done)")

    # Save final results
    out_path = OUT_DIR / "blast_results.json"
    with open(out_path, "w") as f:
        json.dump(blast_results, f, indent=2)
    print(f"\n  Saved: {out_path}")

    total_hits = sum(r["sequences_with_hits"] for r in blast_results.values())
    total_submitted = sum(r["sequences_submitted"] for r in blast_results.values())
    print(f"  Total: {total_hits}/{total_submitted} sequences with hits across {len(blast_results)} latents")

    return blast_results


# ============================================================
#  PART D: ORGANISM LABELING
# ============================================================

def run_part_d(blast_results, enrichment):
    """Label each latent with the organism it detects based on BLAST hits."""
    print(f"\n{'='*60}")
    print(f"PART D: Organism labeling")
    print(f"{'='*60}")

    import csv
    rows = []

    for latent_id, data in blast_results.items():
        lid = int(latent_id)
        hits = data.get("hits", [])

        # Count organisms (genus-level: first two words)
        organisms = Counter()
        identities = []
        evalues = []
        descriptions = []
        n_hits = 0
        n_uncultured = 0
        n_nohit = 0

        for h in hits:
            status = h.get("status", "")
            if status == "hit":
                n_hits += 1
                org_full = h["top_hit"]["organism"]
                org_genus = " ".join(org_full.split()[:2])
                organisms[org_genus] += 1
                identities.append(h["top_hit"].get("percent_identity", 0))
                evalues.append(h["top_hit"].get("e_value", 999))
                descriptions.append(h["top_hit"].get("description", ""))

                # Check for uncultured/environmental
                org_lower = org_full.lower()
                if "uncultured" in org_lower or "environmental" in org_lower:
                    n_uncultured += 1
            elif status == "no_hit":
                n_nohit += 1

        # Labeling logic
        if n_hits == 0:
            label = "uncharacterized"
            confidence = "none"
            consistency = "0/0"
            dominant = ""
            rep_gene = ""
        elif n_uncultured == n_hits:
            label = "uncharacterized environmental"
            confidence = "low"
            dominant = "uncultured/environmental"
            consistency = f"{n_uncultured}/{n_hits}"
            rep_gene = descriptions[0] if descriptions else ""
        else:
            dominant, count = organisms.most_common(1)[0]
            consistency = f"{count}/{n_hits}"

            if count >= 7:
                label = dominant
                confidence = "high"
            elif count >= 5:
                label = dominant
                confidence = "medium"
            else:
                label = "mixed/unresolved"
                confidence = "low"

            # Representative gene: most common description among dominant org hits
            dom_descs = [d for h, d in zip(hits, descriptions)
                         if h.get("status") == "hit" and
                         " ".join(h["top_hit"]["organism"].split()[:2]) == dominant]
            rep_gene = Counter(dom_descs).most_common(1)[0][0] if dom_descs else ""

        mean_ident = np.mean(identities) if identities else 0
        mean_eval = np.mean(evalues) if evalues else 0

        rows.append({
            "latent_id": lid,
            "fisher_or": float(enrichment["fisher_ors"][lid]),
            "fisher_fdr": float(enrichment["fisher_fdrs"][lid]),
            "log2fc": float(enrichment["log2fcs"][lid]),
            "best_f1": float(enrichment["best_f1s"][lid]),
            "dominant_organism": label,
            "hit_consistency": consistency,
            "confidence": confidence,
            "representative_gene": rep_gene,
            "mean_percent_identity": round(mean_ident, 1),
            "mean_e_value": mean_eval,
        })

    # Sort by confidence (high > medium > low > none) then Fisher OR
    conf_order = {"high": 0, "medium": 1, "low": 2, "none": 3}
    rows.sort(key=lambda r: (conf_order.get(r["confidence"], 4), -r["fisher_or"]))

    csv_path = OUT_DIR / "organism_labels.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {csv_path}")

    # Summary
    by_conf = Counter(r["confidence"] for r in rows)
    print(f"\n  SUMMARY:")
    print(f"    High confidence:   {by_conf.get('high', 0)}")
    print(f"    Medium confidence: {by_conf.get('medium', 0)}")
    print(f"    Low confidence:    {by_conf.get('low', 0)}")
    print(f"    Uncharacterized:   {by_conf.get('none', 0)}")

    # Print top results
    high_conf = [r for r in rows if r["confidence"] in ("high", "medium")]
    if high_conf:
        print(f"\n  TOP ORGANISM DETECTORS:")
        print(f"  {'Latent':>8} {'OR':>8} {'F1':>6} {'Organism':>30} {'Consist':>8} {'Identity':>8}")
        for r in high_conf[:15]:
            print(f"  {r['latent_id']:>8} {r['fisher_or']:>8.2f} {r['best_f1']:>6.3f} "
                  f"{r['dominant_organism']:>30} {r['hit_consistency']:>8} "
                  f"{r['mean_percent_identity']:>7.1f}%")
    else:
        print("\n  No high/medium confidence organism labels found.")

    return rows


# ============================================================
#  PART E: FIGURES
# ============================================================

def run_part_e(enrichment, organism_labels=None):
    """Generate volcano plot, bar chart, and enrichment histogram."""
    print(f"\n{'='*60}")
    print(f"PART E: Figures")
    print(f"{'='*60}")

    fisher_ors = enrichment["fisher_ors"]
    fisher_fdrs = enrichment["fisher_fdrs"]
    log2fcs = enrichment["log2fcs"]
    is_path_enriched = enrichment["is_path_enriched"]
    is_nonpath_enriched = enrichment["is_nonpath_enriched"]
    is_path_specific = enrichment["is_path_specific"]

    # --- E1: Volcano plot ---
    fig, ax = plt.subplots(figsize=(10, 7))

    # -log10(FDR), cap at 50 for visualization
    neg_log_fdr = -np.log10(np.clip(fisher_fdrs, 1e-50, 1.0))

    # Colors
    colors = np.full(len(log2fcs), "lightgray", dtype=object)
    colors[is_path_enriched] = "red"
    colors[is_nonpath_enriched] = "blue"

    ax.scatter(log2fcs, neg_log_fdr, c=colors, s=3, alpha=0.4, rasterized=True)

    # Annotate top organism-labeled latents
    if organism_labels:
        top_labeled = [r for r in organism_labels
                       if r["confidence"] in ("high", "medium")][:10]
        for r in top_labeled:
            lid = r["latent_id"]
            ax.annotate(
                f"L{lid}\n{r['dominant_organism']}",
                (log2fcs[lid], neg_log_fdr[lid]),
                fontsize=6, ha="center", va="bottom",
                arrowprops=dict(arrowstyle="-", color="black", lw=0.5),
            )

    ax.axhline(-np.log10(FDR_ALPHA), color="gray", ls="--", lw=0.7, label=f"FDR={FDR_ALPHA}")
    ax.set_xlabel("log2(Fold Change)")
    ax.set_ylabel("-log10(Fisher FDR)")
    ax.set_title("Volcano Plot: SAE Latent Enrichment for Pathogen Detection")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=6, label=f'Pathogen-enriched ({is_path_enriched.sum()})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=6, label=f'Non-pathogen-enriched ({is_nonpath_enriched.sum()})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', markersize=6, label='Not significant'),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "volcano_plot.png", dpi=150)
    plt.close()
    print(f"  Saved: volcano_plot.png")

    # --- E2: Organism detector bar chart ---
    if organism_labels:
        high_conf = [r for r in organism_labels if r["confidence"] in ("high", "medium")][:15]
        if high_conf:
            from matplotlib.patches import Patch
            from collections import defaultdict

            # --- Aggregate by organism ---
            org_data = defaultdict(lambda: {"latents": [], "consistencies": [], "identities": []})
            for r in high_conf:
                org = r["dominant_organism"]
                org_data[org]["latents"].append(r["latent_id"])
                org_data[org]["consistencies"].append(int(r["hit_consistency"].split("/")[0]))
                org_data[org]["identities"].append(float(r.get("mean_percent_identity", 0)))

            # Sort organisms by number of detector latents (descending)
            orgs_sorted = sorted(org_data.keys(), key=lambda o: len(org_data[o]["latents"]), reverse=True)

            # Build summary rows: organism, # latents, mean consistency, mean identity, max consistency
            rows = []
            for org in orgs_sorted:
                d = org_data[org]
                rows.append({
                    "organism": org,
                    "n_latents": len(d["latents"]),
                    "mean_consistency": np.mean(d["consistencies"]),
                    "max_consistency": max(d["consistencies"]),
                    "mean_identity": np.mean(d["identities"]),
                    "best_latent": d["latents"][np.argmax(d["consistencies"])],
                })

            # --- Figure: grouped bar chart ---
            palette = {"Human astrovirus": "#7fb685", "Norovirus GI": "#7b8cc2",
                       "Norovirus GII": "#d4a76a", "Human adenovirus": "#5cb8b2",
                       "Sapovirus GI.1": "#c27b9e", "Astrovirus MLB1": "#b8a9d4",
                       "Mamastrovirus sp.": "#d4c25c"}

            fig, ax = plt.subplots(figsize=(10, 5))

            y_pos = np.arange(len(rows))
            bar_height = 0.6
            colors = [palette.get(r["organism"], "#999999") for r in rows]

            bars = ax.barh(y_pos, [r["n_latents"] for r in rows], height=bar_height,
                           color=colors, edgecolor="white", linewidth=1.2)

            # Labels on bars
            for i, r in enumerate(rows):
                # Number of latents inside the bar
                ax.text(r["n_latents"] - 0.15, i, str(r["n_latents"]),
                        va="center", ha="right", fontsize=11, fontweight="bold", color="white")
                # Annotation to the right
                ax.text(r["n_latents"] + 0.3, i,
                        f"best {r['max_consistency']}/10 consistency  |  {r['mean_identity']:.1f}% avg identity",
                        va="center", ha="left", fontsize=9, color="#444444")

            ax.set_yticks(y_pos)
            ax.set_yticklabels([r["organism"] for r in rows], fontsize=11, fontweight="medium")
            ax.set_xlabel("Number of Detector Latents", fontsize=11)
            ax.set_title("Organism-Specific Pathogen Detectors Identified by BLAST",
                         fontsize=13, fontweight="bold", pad=12)
            ax.invert_yaxis()
            ax.set_xlim(0, max(r["n_latents"] for r in rows) + 8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Subtitle
            n_high = len([r for r in high_conf if r.get("confidence") == "high"])
            n_med = len([r for r in high_conf if r.get("confidence") == "medium"])
            ax.text(0.0, -0.08,
                    f"{n_high} high-confidence + {n_med} medium-confidence detectors across {len(rows)} organisms  |  "
                    f"All detectors fire exclusively on pathogen sequences (Fisher OR = ∞)",
                    transform=ax.transAxes, fontsize=8.5, color="#666666", va="top")

            fig.tight_layout()
            fig.savefig(OUT_DIR / "top_organism_detectors.png", dpi=200, bbox_inches="tight")
            plt.close()
            print(f"  Saved: top_organism_detectors.png")
        else:
            print("  Skipping bar chart: no high/medium confidence labels")

    # --- E3: Enrichment histogram ---
    fig, ax = plt.subplots(figsize=(10, 5))

    # Only alive latents (non-zero, finite OR)
    alive_ors = fisher_ors[(fisher_ors > 0) & np.isfinite(fisher_ors)]
    log2_ors = np.log2(alive_ors)

    ax.hist(log2_ors, bins=100, color="steelblue", edgecolor="none", alpha=0.8)
    ax.axvline(0, color="black", lw=0.8, ls="-")

    # Annotate
    n_pe = is_path_enriched.sum()
    n_npe = is_nonpath_enriched.sum()
    n_ps = is_path_specific.sum()
    ax.text(0.98, 0.95,
            f"{n_pe} pathogen-enriched\n{n_npe} non-pathogen-enriched\n{n_ps} pathogen-specific (F1>{F1_THRESHOLD})",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("log2(Fisher Odds Ratio)")
    ax.set_ylabel("Count (latents)")
    ax.set_title("Distribution of Latent Enrichment for Pathogen Detection")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "enrichment_histogram.png", dpi=150)
    plt.close()
    print(f"  Saved: enrichment_histogram.png")


# ============================================================
#  PART F: API PAYLOAD
# ============================================================

def run_part_f(enrichment, organism_labels=None, blast_results=None, top_sequences=None):
    """Generate api_results.json for the frontend visualizer."""
    print(f"\n{'='*60}")
    print(f"PART F: API payload generation")
    print(f"{'='*60}")

    fisher_ors = enrichment["fisher_ors"]
    fisher_fdrs = enrichment["fisher_fdrs"]
    log2fcs = enrichment["log2fcs"]
    is_path_enriched = enrichment["is_path_enriched"]
    is_nonpath_enriched = enrichment["is_nonpath_enriched"]
    is_path_specific = enrichment["is_path_specific"]
    best_f1s = enrichment["best_f1s"]

    # Summary
    high_conf = [r for r in (organism_labels or []) if r.get("confidence") in ("high",)]
    med_conf = [r for r in (organism_labels or []) if r.get("confidence") in ("medium",)]
    summary = {
        "total_latents": len(fisher_ors),
        "pathogen_enriched": int(is_path_enriched.sum()),
        "nonpathogen_enriched": int(is_nonpath_enriched.sum()),
        "pathogen_specific": int(is_path_specific.sum()),
        "high_confidence_detectors": len(high_conf),
        "medium_confidence_detectors": len(med_conf),
    }

    # Volcano data — subsample ALL categories for Recharts performance (~3k total)
    # Keep all BLAST-analyzed latents, subsample the rest
    blast_latent_ids = set()
    if organism_labels:
        blast_latent_ids = {int(r.get("latent_id", -1)) for r in organism_labels}

    volcano = []
    for i in range(len(fisher_ors)):
        direction = "pathogen" if is_path_enriched[i] else ("nonpathogen" if is_nonpath_enriched[i] else "ns")
        # Always keep BLAST-analyzed latents; subsample rest
        if i not in blast_latent_ids:
            if direction == "ns" and i % 40 != 0:
                continue
            if direction == "pathogen" and i % 8 != 0:
                continue
            if direction == "nonpathogen" and i % 4 != 0:
                continue
        fdr_val = float(fisher_fdrs[i])
        neg_log = min(-np.log10(max(fdr_val, 1e-300)), 50)
        l2fc = float(log2fcs[i])
        if not np.isfinite(l2fc):
            l2fc = 0.0
        if not np.isfinite(neg_log):
            neg_log = 0.0

        entry = {
            "latent_id": int(i),
            "log2fc": round(l2fc, 4),
            "neg_log10_pval": round(neg_log, 4),
            "direction": direction,
        }
        # Add organism label if this latent was BLAST-analyzed
        if organism_labels:
            for r in organism_labels:
                if int(r.get("latent_id", -1)) == i:
                    org = r.get("dominant_organism", "")
                    if org and org not in ("mixed/unresolved", "uncharacterized"):
                        entry["organism"] = org
                    break
        volcano.append(entry)

    # Detectors — the 50 BLAST-analyzed latents with full hit data
    detectors = []
    if organism_labels and blast_results and top_sequences:
        for r in organism_labels:
            lid = str(r.get("latent_id", ""))
            blast_data = blast_results.get(lid, {})
            seq_data = top_sequences.get(lid, {})
            hits = blast_data.get("hits", [])

            blast_hits = []
            top_seqs = {s["sequence_id"]: s for s in seq_data.get("top_sequences", [])}
            for h in hits:
                if h.get("status") != "hit":
                    continue
                th = h.get("top_hit", {})
                blast_hits.append({
                    "sequence_id": h.get("sequence_id", ""),
                    "organism": th.get("organism", ""),
                    "accession": th.get("accession", ""),
                    "percent_identity": th.get("percent_identity", 0),
                    "e_value": th.get("e_value", 0),
                    "bit_score": th.get("bit_score", 0),
                    "sequence": top_seqs.get(h.get("sequence_id", ""), {}).get("sequence", ""),
                })

            fisher_or_val = r.get("fisher_or", 0)
            if isinstance(fisher_or_val, str):
                try:
                    fisher_or_val = float(fisher_or_val)
                except (ValueError, TypeError):
                    fisher_or_val = 0
            if not np.isfinite(fisher_or_val):
                fisher_or_val = 999.0  # cap inf for JSON

            detectors.append({
                "latent_id": int(r.get("latent_id", 0)),
                "fisher_or": round(float(fisher_or_val), 2),
                "log2fc": round(float(r.get("log2fc", 0)), 4),
                "best_f1": round(float(r.get("best_f1", 0)), 4),
                "dominant_organism": r.get("dominant_organism", ""),
                "hit_consistency": r.get("hit_consistency", ""),
                "confidence": r.get("confidence", ""),
                "mean_percent_identity": round(float(r.get("mean_percent_identity", 0)), 1),
                "blast_hits": blast_hits,
            })

    # Enrichment histogram
    finite_ors = fisher_ors[(fisher_ors > 0) & np.isfinite(fisher_ors)]
    log2_ors = np.log2(finite_ors)
    counts, edges = np.histogram(log2_ors, bins=40)
    enrichment_histogram = [
        {"bin_start": round(float(edges[i]), 3), "bin_end": round(float(edges[i+1]), 3), "count": int(counts[i])}
        for i in range(len(counts))
    ]

    payload = {
        "summary": summary,
        "volcano": volcano,
        "detectors": detectors,
        "enrichment_histogram": enrichment_histogram,
    }

    out_path = OUT_DIR / "api_results.json"
    with open(out_path, "w") as f:
        json.dump(payload, f)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {out_path} ({size_mb:.1f} MB)")
    print(f"  Volcano points: {len(volcano)}, Detectors: {len(detectors)}")


# ============================================================
#  MAIN
# ============================================================

def main():
    global DATA_DIR, OUT_DIR
    parser = argparse.ArgumentParser(description="Experiment 1: Organism-Specific Pathogen Detectors")
    parser.add_argument("--parts", default="ABCDEF",
                        help="Which parts to run (e.g., 'AB', 'C', 'DEF', 'ABCDEF')")
    parser.add_argument("--blast-test", action="store_true",
                        help="Test BLAST with 3 latents x 3 sequences only")
    parser.add_argument("--layer", type=int, default=32,
                        help="Which layer's SAE features to use (8, 16, 24, 32)")
    args = parser.parse_args()

    # Set data/output dirs based on layer
    if args.layer in LAYER_DATA_DIRS:
        DATA_DIR = LAYER_DATA_DIRS[args.layer]
    else:
        DATA_DIR = Path(f"data/sae_layer{args.layer}")
    if args.layer == 32:
        OUT_DIR = Path("results/organism_detectors")
    else:
        OUT_DIR = Path(f"results/organism_detectors_layer{args.layer}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    parts = args.parts.upper()
    print(f"Running parts: {parts}")
    print(f"Layer: {args.layer}")
    print(f"Data dir: {DATA_DIR}")
    print(f"Output dir: {OUT_DIR}")
    print(f"BLAST test mode: {args.blast_test}")

    features = None
    sequence_ids = None
    labels = None
    pathogen = None
    seq_lookup = None
    enrichment = None
    top_sequences = None
    blast_results = None
    organism_labels = None

    # Load data if needed for any part
    if any(p in parts for p in "AB"):
        features, sequence_ids, labels, pathogen, seq_lookup = load_data()

    # Part A
    if "A" in parts:
        enrichment = run_part_a(features, pathogen)

    # Part B
    if "B" in parts:
        if enrichment is None:
            enrichment = load_enrichment_from_csv()
        if features is None:
            features, sequence_ids, labels, pathogen, seq_lookup = load_data()

        if args.blast_test:
            top_sequences = run_part_b(features, sequence_ids, pathogen, seq_lookup,
                                       enrichment, max_latents=3, max_seqs=3)
        else:
            top_sequences = run_part_b(features, sequence_ids, pathogen, seq_lookup,
                                       enrichment)

    # Part C
    if "C" in parts:
        if top_sequences is None:
            seq_path = OUT_DIR / "top_sequences_per_latent.json"
            if not seq_path.exists():
                print(f"ERROR: {seq_path} not found. Run Part B first.")
                sys.exit(1)
            with open(seq_path) as f:
                top_sequences = json.load(f)

        blast_results = run_part_c(top_sequences, blast_test=args.blast_test)

    # Part D
    if "D" in parts:
        if blast_results is None:
            blast_path = OUT_DIR / "blast_results.json"
            if not blast_path.exists():
                print(f"ERROR: {blast_path} not found. Run Part C first.")
                sys.exit(1)
            with open(blast_path) as f:
                blast_results = json.load(f)
        if enrichment is None:
            enrichment = load_enrichment_from_csv()

        organism_labels = run_part_d(blast_results, enrichment)

    # Part E
    if "E" in parts:
        if enrichment is None:
            enrichment = load_enrichment_from_csv()

        # Try to load organism labels if we have them
        if organism_labels is None:
            labels_path = OUT_DIR / "organism_labels.csv"
            if labels_path.exists():
                import csv
                with open(labels_path) as f:
                    reader = csv.DictReader(f)
                    organism_labels = []
                    for row in reader:
                        row["latent_id"] = int(row["latent_id"])
                        row["fisher_or"] = float(row["fisher_or"])
                        organism_labels.append(row)

        run_part_e(enrichment, organism_labels)

    # Part F
    if "F" in parts:
        if enrichment is None:
            enrichment = load_enrichment_from_csv()
        if organism_labels is None:
            labels_path = OUT_DIR / "organism_labels.csv"
            if labels_path.exists():
                import csv
                with open(labels_path) as f:
                    reader = csv.DictReader(f)
                    organism_labels = [dict(row) for row in reader]
                    for row in organism_labels:
                        row["latent_id"] = int(row["latent_id"])
        if blast_results is None:
            blast_path = OUT_DIR / "blast_results.json"
            if blast_path.exists():
                with open(blast_path) as f:
                    blast_results = json.load(f)
        if top_sequences is None:
            seq_path = OUT_DIR / "top_sequences_per_latent.json"
            if seq_path.exists():
                with open(seq_path) as f:
                    top_sequences = json.load(f)
        run_part_f(enrichment, organism_labels, blast_results, top_sequences)

    print(f"\n{'='*60}")
    print(f"DONE. Results in {OUT_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
