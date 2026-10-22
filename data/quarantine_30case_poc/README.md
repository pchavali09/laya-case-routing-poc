# Quarantined 30-Case Mock POC Files

These files contain the legacy 30-case synthetic dataset, splits (including simulated Spanish/Hindi cases), and mock output files from the initial proof-of-concept.

They have been quarantined because:
1. They used simulated/hardcoded accuracy figures for Einstein (73.3%) rather than measured Salesforce sandbox results.
2. The benchmark evaluation has been superseded by the 1,000-case real Salesforce Einstein benchmark in `data/einstein_benchmark/` (800 train, 200 blind test, evaluated against ground truth and real Salesforce Einstein predictions).
