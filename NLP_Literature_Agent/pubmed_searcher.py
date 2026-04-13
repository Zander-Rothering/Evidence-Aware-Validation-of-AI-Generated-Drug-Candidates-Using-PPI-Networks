"""PubMed Literature Searcher — Fetch paper abstracts via NCBI Entrez.

Not yet fully implemented. Hardcoded now for `ner_extractor.py` quick test.
"""


class PubMedSearcher:
    """Search PubMed and retrieve abstracts using NCBI Entrez."""

    def __init__(self, max_results: int = 20, email: str = "") -> None:
        from Bio import Entrez
        Entrez.email = email
        self.max_results = max_results
        self.entrez = Entrez

    def search(self, search_terms: str, sider_risks: list[str] = None, target: str = "HMGCR") -> dict:
        # 4 fallback query levels, from most specific to least
        queries = [
            f"{search_terms} {target}",                         # level 0: compound + target
            f"{search_terms} statin HMGCR inhibitor",           # level 1: scaffold + target
            f"statin HMGCR inhibitor safety adverse effects",   # level 2: drug class + target
            f"HMGCR inhibitor adverse effects",                 # level 3: target only
        ]
        


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
    print("PubMedSearcher — scaffold (not yet implemented)")
    print(f"Sample abstracts: {len(SAMPLE_RESULT['abstracts'])}")
    for i, abstract in enumerate(SAMPLE_RESULT["abstracts"], 1):
        print(f"\n--- Abstract {i} ---")
        print(abstract)
