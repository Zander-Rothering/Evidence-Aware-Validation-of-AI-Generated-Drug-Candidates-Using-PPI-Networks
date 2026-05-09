"""PubMed Literature Searcher — Fetch paper abstracts via NCBI Entrez.

Queries PubMed using a 4-level fallback strategy:
  Level 0: compound name + target (most specific)
  Level 1: compound + statin HMGCR inhibitor
  Level 2: drug class + target
  Level 3: target only (broadest fallback)

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

    def __init__(self, max_results: int = 20, email: str = "example@berkeley.edu") -> None:
        Entrez.email = email            # NCBI wants a contact email on every Entrez request
        self.max_results = max_results  # cap PMID count per esearch (also limits efetch batch size)
        self.entrez = Entrez            # Bio.Entrez module (esearch / efetch entry points)

    def search(self, search_terms: str, sider_risks: list[str] = None, target: str = "HMGCR") -> dict:
        # 4 fallback query levels, from most specific to least
        queries = [
            f"{search_terms} {target}",                         # level 0: compound + target
            f"{search_terms} statin HMGCR inhibitor",           # level 1: scaffold + target
            f"statin HMGCR inhibitor safety adverse effects",   # level 2: drug class + target
            f"HMGCR inhibitor adverse effects",                 # level 3: target only
        ]

        # if `sider_risks` exists, add them to the query at level 1.5
        if sider_risks:
            risk_str = " ".join(sider_risks[:3])
            queries.insert(2, f"{search_terms} {risk_str}")      # level 1.5: compound + known risks

        # Try each query: narrow first, then broader fallbacks until we get abstracts
        for level, query in enumerate(queries):
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
                handle = self.entrez.efetch(
                    db="pubmed",
                    id=",".join(pmids),
                    rettype="medline",
                    retmode="text"
                )
                content = handle.read()
                # BioPython may return str or bytes depending on version / handle
                raw = content.decode(errors="ignore") if isinstance(content, bytes) else content
                handle.close()

                # Split on Medline "AB  - " (abstract field); drop chunks that are too short
                abstracts = [chunk.strip() for chunk in raw.split("AB  - ")[1:] if len(chunk.strip()) > 50]

                # At least one usable abstract -> success; exit search() with this level tag
                if abstracts:
                    return {"pmids": pmids, "abstracts": abstracts, "evidence_source": f"level_{level}"}

            # Network / NCBI / parse error at this level -> try the next fallback query
            except Exception as e:
                print(f"[PubMedSearcher] level {level} failed: {e!r}")
                traceback.print_exc()
                continue

        # All query levels exhausted (no hits or no usable abstracts) -> hardcoded sample for NER/offline
        return SAMPLE_RESULT


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
    "evidence_source": "level_0",  # which fallback level found results
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
