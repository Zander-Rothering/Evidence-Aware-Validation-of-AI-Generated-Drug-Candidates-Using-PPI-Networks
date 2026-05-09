"""PubMed Literature Searcher — Fetch paper abstracts via NCBI Entrez.

Queries PubMed using a 5-level fallback strategy:
  Level 0: compound name + target (most specific)
  Level 1: compound + target + safety
  Level 2: compound + top SIDER safety cues (max 3 terms)
  Level 3: statin class + target + adverse effects
  Level 4: target only (broadest fallback)

Returns pmids, abstracts, and the fallback level used.
Falls back to `SAMPLE_RESULT` if all API calls fail.

Requirements:
  - biopython >= 1.87
  - Python 3.9-3.12 recommended
    (Python 3.13 on macOS may cause SSL certificate errors
     with NCBI Entrez; use a conda environment to avoid this)

Developed with AI assistance (Claude, Anthropic) for syntax support.
"""

import traceback
from Bio import Entrez


class PubMedSearcher:
    """Search PubMed and retrieve abstracts using NCBI Entrez."""

    def __init__(self, max_results: int = 10, email: str = "example@berkeley.edu") -> None:
        Entrez.email = email            # NCBI wants a contact email on every Entrez request
        self.max_results = max_results  # cap PMID count per esearch (also limits efetch batch size)
        self.entrez = Entrez            # Bio.Entrez module (esearch / efetch entry points)

    PRIORITY_SAFETY_TERMS = [
        "myopathy",
        "rhabdomyolysis",
        "hepatotoxicity",
        "liver",
        "muscle",
    ]

    def top_safety_terms(self, sider_risks: list[str] | None, max_terms: int = 3) -> list[str]:
        """Pick a small set of high-value safety cues from SIDER terms."""
        if not sider_risks:
            return []
        normalized = list(dict.fromkeys(text for text in (str(term).strip().lower() for term in sider_risks) if text))
        priority_hits = [term for term in self.PRIORITY_SAFETY_TERMS if term in normalized]
        remaining = [term for term in normalized if term not in priority_hits]
        picked = priority_hits + remaining
        return picked[:max_terms]

    @staticmethod
    def compact(text: str, max_chars: int = 140) -> str:
        value = " ".join(str(text).split())
        return value[:max_chars].strip()

    def build_queries(self, search_terms: str, sider_risks: list[str] | None, target: str, max_queries: int = 5) -> list[tuple[str, str]]:
        """Build PubMed fallback queries with explicit evidence levels."""
        seed = self.compact(search_terms or "")
        target_term = self.compact(target or "HMGCR")
        safety_terms = self.top_safety_terms(sider_risks, max_terms=3)
        safety_str = " ".join(safety_terms)
        seed_tokens = [tok for tok in seed.split() if tok.lower() != target_term.lower()]
        seed_no_target = " ".join(seed_tokens)
        is_chembl_seed = seed_no_target.lower().startswith("chembl")

        candidates: list[tuple[str, str]] = []
        if not is_chembl_seed:
            candidates.extend([("level_0", f"{seed_no_target} {target_term}".strip()), ("level_1", f"{seed_no_target} {target_term} safety".strip())])
            if safety_str:
                candidates.append(("level_2", f"{seed_no_target} {safety_str}".strip()))
        candidates.extend([("level_3", "statin HMGCR inhibitor adverse effects"), ("level_4", "HMGCR inhibitor adverse effects")])
        # de-duplicate and drop empties while preserving order
        out: list[tuple[str, str]] = []
        seen = set()
        for evidence_source, query in candidates:
            query = self.compact(query, max_chars=180)
            if not query:
                continue
            key = query.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append((evidence_source, query))
            if len(out) >= max_queries:
                break
        return out[:max(2, min(max_queries, len(candidates)))]

    def search(self, search_terms: str, sider_risks: list[str] = None, target: str = "HMGCR") -> dict:
        # Fallback query levels, from specific seed to broad class query.
        queries = self.build_queries(search_terms=search_terms, sider_risks=sider_risks, target=target, max_queries=5)

        # Try each query: narrow first, then broader fallbacks until we get abstracts
        for evidence_source, query in queries:
            try:
                # NCBI esearch: PubMed IDs matching this query (up to max_results)
                handle = self.entrez.esearch(db="pubmed", term=query, retmax=self.max_results)
                record = self.entrez.read(handle)
                handle.close()
                pmids = record.get("IdList", [])

                # No hits at this level -> skip to the next (broader) query
                if not pmids:
                    continue

                # efetch: full medline text for those PMIDs (comma-separated for NCBI)
                handle = self.entrez.efetch(db="pubmed", id=",".join(pmids), rettype="medline", retmode="text")

                content = handle.read()
                # BioPython may return str or bytes depending on version / handle
                raw = content.decode(errors="ignore") if isinstance(content, bytes) else content
                handle.close()

                # Split on Medline "AB  - " (abstract field); drop chunks that are too short
                abstracts = [chunk.strip() for chunk in raw.split("AB  - ")[1:] if len(chunk.strip()) > 50]

                # At least one usable abstract -> success; exit search() with this level tag
                if abstracts:
                    return {"pmids": pmids, "abstracts": abstracts, "evidence_source": evidence_source}

            # Network / NCBI / parse error at this level -> try the next fallback query
            except Exception as e:
                print(f"[PubMedSearcher] {evidence_source} failed: {e!r}")
                traceback.print_exc()
                continue

        # All query levels exhausted (no hits or no usable abstracts) -> hardcoded sample for NER/offline
        return dict(SAMPLE_RESULT)


# Hard_coded quick test data so we can try ner_extractor without calling the real API
SAMPLE_RESULT = {
    #https://pubmed.ncbi.nlm.nih.gov/17910522/
    "pmids": ["17910522"],  # real PubMed ID: "Atorvastatin: a safety and tolerability profile"
    "abstracts": [
        "Extensive data are available on the safety of atorvastatin from randomised "
        "clinical trials, postmarketing analyses and reports to regulatory agencies. "
        "Atorvastatin is generally well tolerated across the range of therapeutic "
        "dosages, with the exception of a slightly higher rate of liver enzyme "
        "elevations with atorvastatin 80 mg/day which does not appear to confer an "
        "increased risk of clinically important adverse events. Unlike simvastatin, "
        "atorvastatin is associated with a low incidence of muscular toxicity. It is "
        "not associated with neurological, cognitive or renal adverse effects and does "
        "not require dosage adjustment in patients with renal dysfunction, due to its "
        "favourable pharmacokinetic profile, which is unique among the statins. In "
        "patients aged > or =65 years, atorvastatin is well tolerated with no "
        "dose-dependent increase in adverse events up to the maximum daily dosage of "
        "80 mg/day. Thus, atorvastatin is a safe and well tolerated statin for use in "
        "a wide range of patients.",
    ],
    "evidence_source": "sample_fallback",  # synthetic fallback, not a real PubMed hit
}


if __name__ == "__main__":
    # Runs search(); output is live PubMed unless all levels fail (then SAMPLE_RESULT).
    searcher = PubMedSearcher(max_results=3, email="example@berkeley.edu")
    search_result = searcher.search("atorvastatin")
    print("evidence_source:", search_result.get("evidence_source"))
    print("pmids:", search_result.get("pmids"))
    print("n_abstracts:", len(search_result.get("abstracts", [])))
    for i, abstract_text in enumerate(search_result.get("abstracts", []), 1):
        print(f"\n--- Abstract {i} ---")
        print(abstract_text[:800] + ("..." if len(abstract_text) > 800 else ""))
