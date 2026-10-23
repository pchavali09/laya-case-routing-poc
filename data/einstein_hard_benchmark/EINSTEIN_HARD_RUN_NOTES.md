# Einstein run on the hard benchmark (facts reported by Pavan, not measured by code)

- Run date: 2026-10-23 (sandbox). Model name in setup: "Case Classification POC".
- Training data: data/einstein_hard_benchmark/train_salesforce_import.csv, 800 Closed cases (HARD-TRAIN-*).
  Verified in the org by SOQL: 160 per department, all Closed. Original ENS-* cases deleted first (count = 0).
- Setup: field Department__c; Select Value = "Select Best", threshold 0; Automate Value = On, confidence threshold 90.
- Test data: test_salesforce_import.csv (200, HARD-TEST-*), imported AFTER the model was activated.
- Einstein confidence is not exposed in this org. A blank Department__c means not routed; a filled one means routed.
- Provenance check (SOQL): HARD-TEST-0002 and -0003 were last modified by "Automated Process" 12 and 15 s after creation;
  HARD-TEST-0001 and -0004 (blank) were not modified after import. No field history on Department__c; no default value.
- Raw export: einstein_hard_raw_export.csv (columns: _, ExternalId__c, Department__c, CreatedDate). Do not edit it.
- Not run: Einstein trained on the hard 800 scored on the original 200 test set.
- Caveat: the training set has ~132 distinct descriptions (digits masked); ~20 themes x 40 near-copies.
