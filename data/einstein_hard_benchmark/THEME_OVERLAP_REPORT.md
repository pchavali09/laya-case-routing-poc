# Group-Held-Out Benchmark: Theme & Overlap Audit Report

**Audit Date:** 2026-10-22  
**Dataset Directory:** `data/einstein_hard_benchmark`  
**Split Method:** Group-held-out template families (themes). Whole themes are held out exclusively in test.

---

## 1. Executive Summary

| Metric | Result | Target / Standard | Status |
| :--- | :--- | :--- | :--- |
| **Shared Themes** | **0** (0 / 30 total themes) | 0 | ✅ PASS (Zero Overlap) |
| **Exact Full-Text Matches** | **0** (0 / 200 cases) | 0 | ✅ PASS |
| **Exact Subject Matches** | **0** (0 / 200 cases) | 0 | ✅ PASS |
| **Vocabulary Jaccard Similarity** | **25.87%** | < 40% | ✅ PASS |
| **2-Gram Jaccard Similarity** | **3.49%** | < 15% | ✅ PASS |
| **3-Gram Jaccard Similarity** | **0.52%** | < 5% | ✅ PASS |
| **4-Gram Jaccard Similarity** | **0.02%** | < 2% | ✅ PASS |
| **Mean Max TF-IDF Sim to Train** | **0.1578** | Moderate lexical distance | ✅ Verified |
| **Median Max TF-IDF Sim to Train** | **0.1505** | Moderate lexical distance | ✅ Verified |
| **Highest Max TF-IDF Sim to Train** | **0.4315** | No near-duplicates (< 0.65) | ✅ Verified |

---

## 2. Department & Theme Allocation Breakdown

### Training Set (800 Cases, 20 Themes, 4 Themes/Dept, 40 Cases/Theme)
- **Billing** (160 cases):
  1. `invoice_line_items_and_statements`
  2. `tax_exemption_and_vat_compliance`
  3. `gateway_and_ach_failures`
  4. `license_seat_proration_and_credits`
- **Technical** (160 cases):
  1. `database_performance_and_latency`
  2. `sso_and_saml_authentication`
  3. `api_500_and_payload_errors`
  4. `container_and_pod_crashes`
- **Sales** (160 cases):
  1. `product_demos_and_poc_evaluations`
  2. `pricing_schedules_and_volume_quotes`
  3. `partner_and_reseller_agreements`
  4. `annual_contract_renewals`
- **Operations** (160 cases):
  1. `facilities_and_hvac_controls`
  2. `physical_security_and_rfid_badges`
  3. `hardware_asset_tagging_and_procurement`
  4. `office_moves_and_desk_reconfiguration`
- **Other** (160 cases):
  1. `data_privacy_and_gdpr_deletion`
  2. `legal_notices_and_dmca`
  3. `marketing_communications_and_unsubscribes`
  4. `recruiting_and_employment_verification`

---

### Held-Out Test Set (200 Cases, 10 Themes, 2 Themes/Dept, 20 Cases/Theme)
*Note: None of the following themes appear in the 800 training cases.*

- **Billing** (40 cases):
  1. `unauthorized_banking_withdrawals_and_disputes` *(Chargebacks, unauthorized direct debits, banking disputes)*
  2. `currency_conversion_and_cross_border_fees` *(FX rates, ECB benchmarks, cross-border withholding tax)*
- **Technical** (40 cases):
  1. `client_sdk_exceptions_and_syntax_bugs` *(Python/Java/Go SDK runtime panics, TypeError, NullPointer)*
  2. `streaming_rate_limits_and_websockets` *(WebSocket drops, HTTP/2 window scaling, gRPC backpressure, 429)*
- **Sales** (40 cases):
  1. `competitor_takeover_and_migration_buyouts` *(Zendesk/ServiceNow displacement, vendor buyout credits)*
  2. `custom_sla_and_msa_legal_redlines` *(99.99% penalty SLAs, HIPAA BAA clauses, MSA indemnification)*
- **Operations** (40 cases):
  1. `critical_power_and_generator_systems` *(Diesel generator transfer switch failure, UPS battery thermal run)*
  2. `international_freight_and_customs_logistics` *(Air freight customs delays, Bill of Lading, damaged pallets)*
- **Other** (40 cases):
  1. `academic_research_and_educational_grants` *(University AI lab collaborations, collegiate grant applications)*
  2. `corporate_social_responsibility_and_charity` *(Diversity conference sponsorships, 501(c)(3) matching)*

---

## 3. Lexical and Semantic Separation Analysis

When synthetic datasets share template families, simple n-gram memorizers (such as TF-IDF) achieve artificially inflated scores (e.g., 98.5%) because sentence structures repeat. 

In this group-held-out benchmark:
1. **0% Theme Overlap**: The model is evaluated on concepts it has never seen in training (e.g. diesel generator failures in Operations; client SDK TypeErrors in Technical).
2. **Low 3-Gram Overlap (0.52%)**: Guarantees that models cannot rely on memorized multi-word template fragments.
3. **Max TF-IDF Cosine Similarity (0.4315)**: Confirms that test cases represent distinct semantic instances rather than paraphrased variations of training items.
