# Turnkey Instructions for Scoring Hard Benchmark in Salesforce Einstein Sandbox

**Attention:** Pavan Chavali  
**Target File for Import:** [`data/einstein_hard_benchmark/test_salesforce_import.csv`](file:///Users/pavanchavali/Documents/SFProjects/sf-claude-test/laya-case-routing-poc/data/einstein_hard_benchmark/test_salesforce_import.csv)  
**Destination Output File:** `data/einstein_hard_benchmark/einstein_predictions.csv`  

> [!IMPORTANT]
> **Zero Invented Numbers Policy:** We do NOT mock or simulate Einstein predictions. Until you run this import and export the real scores from your sandbox, all benchmark tables will report Einstein on the hard test set as `Pending Sandbox Export`.

---

## Step 1: Import the 200 Test Cases into Sandbox

1. Open your Salesforce Sandbox.
2. Launch **Salesforce Data Loader** or navigate to **Setup -> Data Management -> Data Import Wizard**.
3. Select **Standard Objects -> Cases -> Add New Records**.
4. Choose the file:  
   `data/einstein_hard_benchmark/test_salesforce_import.csv`
5. Map the CSV fields to Case fields:
   - `ExternalId__c` &rarr; `Case.ExternalId__c` (Set as external ID)
   - `Subject` &rarr; `Case.Subject`
   - `Description` &rarr; `Case.Description`
   - `Origin` &rarr; `Case.Origin`
   - `Priority` &rarr; `Case.Priority`
   - `Status` &rarr; `Case.Status` (values are `New`)
6. Execute the import (200 records should succeed with 0 errors).

---

## Step 2: Trigger Einstein Case Classification

1. Ensure the active Einstein Case Classification model in your sandbox is configured to evaluate incoming `New` cases on `Subject` and `Description` to predict `Department__c`.
2. If your model evaluates automatically upon case creation, verify that the 200 cases received Einstein recommendations.
3. If recommendations require batch execution or automated flow execution, trigger the flow for cases where `ExternalId__c LIKE 'HARD-TEST-%'`.

---

## Step 3: Export the Real Predictions

Execute the following SOQL query in Developer Console, Salesforce Inspector, or via Salesforce CLI:

```sql
SELECT 
    ExternalId__c,
    Department__c,
    Einstein_Department_Prediction__c,
    Einstein_Department_Confidence__c,
    Is_Routed_By_Einstein__c
FROM Case 
WHERE ExternalId__c LIKE 'HARD-TEST-%'
ORDER BY ExternalId__c ASC
```

*(Note: Use whatever exact custom field API names your org uses for Einstein's recommendation and confidence score).*

---

## Step 4: Save the Export to the Repository

Save the exported CSV to:  
`data/einstein_hard_benchmark/einstein_predictions.csv`

Ensure the final CSV has these standard header names:
```csv
ExternalId__c,predicted_department,confidence,is_routed
HARD-TEST-0001,Technical,0.88,True
HARD-TEST-0002,Other,0.91,True
...
```
Where:
- `is_routed`: `True` if Einstein met its automated routing threshold (or if the confidence exceeded your org's automation threshold, typically 0.85/0.75), `False` if flagged for human review.

Once placed in `data/einstein_hard_benchmark/einstein_predictions.csv`, running `make benchmark` will automatically compute and report Einstein's official accuracy and coverage on the hard benchmark!
