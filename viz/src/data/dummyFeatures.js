// Dummy data — replace with real SAE feature data when available

function makeTokens(seq, activationRegion) {
  return seq.split("").map((ch, i) => ({
    text: ch,
    activation:
      i >= activationRegion[0] && i < activationRegion[1]
        ? 0.4 + Math.random() * 0.6
        : Math.random() * 0.08,
  }));
}

function makeHistogram() {
  const bins = 20;
  const vals = [];
  for (let i = 0; i < bins; i++) {
    vals.push(Math.exp(-i * 0.3) * (0.6 + Math.random() * 0.4));
  }
  const max = Math.max(...vals);
  return vals.map((v) => v / max);
}

export const DUMMY_FEATURES = [
  {
    id: 127,
    label: "Viral capsid motif",
    description:
      "Activates strongly on sequences containing conserved viral capsid protein signatures. Enriched in human-infecting RNA viruses.",
    layer: 16,
    freqActive: 0.034,
    maxAct: 8.42,
    category: "Short Motif",
    activationPattern: "Short Motif",
    histogram: makeHistogram(),
    topTaxa: [
      { name: "Influenza A", score: 0.82 },
      { name: "SARS-CoV-2", score: 0.71 },
      { name: "Rhinovirus", score: 0.45 },
      { name: "Norovirus", score: 0.31 },
      { name: "Rotavirus", score: 0.18 },
    ],
    topSequences: [
      {
        source: "human_virus_class1",
        classLabel: "Class 1",
        maxAct: 8.42,
        tokens: makeTokens(
          "ATGCGTACGATCGATCGTAGCTAGCTGATCGATCGATCGTAGCTAGCTGA",
          [12, 28]
        ),
      },
      {
        source: "human_virus_class2",
        classLabel: "Class 2",
        maxAct: 7.91,
        tokens: makeTokens(
          "GCTAGCTAGCGATCGATCGATCGTAGCTAGCTGATCGATCGATCGTAGCT",
          [10, 26]
        ),
      },
      {
        source: "human_virus_class1",
        classLabel: "Class 1",
        maxAct: 6.55,
        tokens: makeTokens(
          "TGATCGATCGTAGCTAGCTGATCGATCGATCGTAGCTAGCTGATCGATCG",
          [8, 24]
        ),
      },
    ],
  },
  {
    id: 842,
    label: "Bacterial 16S signature",
    description:
      "Fires on conserved regions of 16S ribosomal RNA sequences. Strongest for gram-negative bacteria in wastewater samples.",
    layer: 20,
    freqActive: 0.087,
    maxAct: 12.1,
    category: "Domain",
    activationPattern: "Domain",
    histogram: makeHistogram(),
    topTaxa: [
      { name: "E. coli", score: 0.91 },
      { name: "Klebsiella", score: 0.78 },
      { name: "Pseudomonas", score: 0.65 },
      { name: "Salmonella", score: 0.52 },
      { name: "Bacteroides", score: 0.41 },
    ],
    topSequences: [
      {
        source: "hmpd_source",
        classLabel: "Bacteria",
        maxAct: 12.1,
        tokens: makeTokens(
          "AGAGTTTGATCCTGGCTCAGATTGAACGCTGGCGGCATGCCTAACACATG",
          [0, 35]
        ),
      },
      {
        source: "hmpd_disease",
        classLabel: "Bacteria",
        maxAct: 10.8,
        tokens: makeTokens(
          "CCTGGCTCAGATTGAACGCTGGCGGCATGCCTAACACATGCAAGTCGAAC",
          [0, 38]
        ),
      },
      {
        source: "hmpd_source",
        classLabel: "Bacteria",
        maxAct: 9.32,
        tokens: makeTokens(
          "TGATCCTGGCTCAGATTGAACGCTGGCGGCATGCCTAACACATGCAAGTC",
          [0, 32]
        ),
      },
    ],
  },
  {
    id: 2041,
    label: "GC-rich region detector",
    description:
      "Responds to high GC-content stretches. May correspond to thermophilic organism sequences or specific gene regulatory regions.",
    layer: 12,
    freqActive: 0.142,
    maxAct: 5.67,
    category: "Whole",
    activationPattern: "Whole",
    histogram: makeHistogram(),
    topTaxa: [
      { name: "Thermus thermophilus", score: 0.73 },
      { name: "Deinococcus radiodurans", score: 0.61 },
      { name: "Streptomyces", score: 0.54 },
      { name: "Mycobacterium", score: 0.38 },
      { name: "Actinobacteria sp.", score: 0.25 },
    ],
    topSequences: [
      {
        source: "hmpd_source",
        classLabel: null,
        maxAct: 5.67,
        tokens: makeTokens(
          "GCCGCGCCGCGCCGCGGCCGCGCCGCCGCGCCGCGCCGCGGCCGCGCCG",
          [0, 50]
        ),
      },
      {
        source: "hvr_default",
        classLabel: null,
        maxAct: 4.89,
        tokens: makeTokens(
          "CCGCGGCCGCGCCGCCGCGCCGCGCCGCGGCCGCGCCGCCGCGCCGCGCC",
          [0, 50]
        ),
      },
      {
        source: "hmpd_disease",
        classLabel: null,
        maxAct: 4.21,
        tokens: makeTokens(
          "GCGCCGCGGCCGCGCCGCCGCGCCGCGCCGCGGCCGCGCCGCCGCGCCGC",
          [0, 50]
        ),
      },
    ],
  },
  {
    id: 3599,
    label: "Phage integrase site",
    description:
      "Activates at attachment sites (attP/attB) associated with temperate phage integration. Point-like activation.",
    layer: 24,
    freqActive: 0.012,
    maxAct: 15.3,
    category: "Point",
    activationPattern: "Point",
    histogram: makeHistogram(),
    topTaxa: [
      { name: "Lambda phage", score: 0.88 },
      { name: "P22 phage", score: 0.72 },
      { name: "Mu phage", score: 0.44 },
      { name: "T4 phage", score: 0.15 },
    ],
    topSequences: [
      {
        source: "human_virus_class1",
        classLabel: "Phage",
        maxAct: 15.3,
        tokens: makeTokens(
          "ATCGATCGATCGATCGCTTTGCATTAGCTGATCGATCGATCGATCGATCG",
          [18, 20]
        ),
      },
      {
        source: "human_virus_class2",
        classLabel: "Phage",
        maxAct: 13.7,
        tokens: makeTokens(
          "GATCGATCGATCGCTTTGCATTAGCTGATCGATCGATCGATCGATCGATC",
          [17, 19]
        ),
      },
      {
        source: "human_virus_class1",
        classLabel: "Phage",
        maxAct: 11.2,
        tokens: makeTokens(
          "TCGATCGATCGATCGCTTTGCATTAGCTGATCGATCGATCGATCGATCGA",
          [19, 21]
        ),
      },
    ],
  },
  {
    id: 1456,
    label: "Human microbiome core",
    description:
      "Broadly activated across human gut microbiome samples. Low specificity but high frequency — may represent a 'background' feature.",
    layer: 16,
    freqActive: 0.312,
    maxAct: 3.21,
    category: "Whole",
    activationPattern: "Whole",
    histogram: makeHistogram(),
    topTaxa: [
      { name: "Bacteroides fragilis", score: 0.55 },
      { name: "Faecalibacterium", score: 0.51 },
      { name: "Prevotella", score: 0.48 },
      { name: "Ruminococcus", score: 0.42 },
      { name: "Bifidobacterium", score: 0.39 },
    ],
    topSequences: [
      {
        source: "hmpd_source",
        classLabel: "Gut",
        maxAct: 3.21,
        tokens: makeTokens(
          "ATGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATC",
          [0, 50]
        ),
      },
      {
        source: "hmpd_disease",
        classLabel: "Gut",
        maxAct: 2.98,
        tokens: makeTokens(
          "GATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGA",
          [0, 50]
        ),
      },
      {
        source: "hmpd_sex",
        classLabel: "Gut",
        maxAct: 2.74,
        tokens: makeTokens(
          "TCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATC",
          [0, 50]
        ),
      },
    ],
  },
];
