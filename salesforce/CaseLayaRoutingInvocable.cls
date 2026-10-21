/**
 * @description Invocable Apex Action that calls the local-first Laya Case Routing Endpoint
 * Can be called from Salesforce Flow, Process Builder, or Apex Trigger.
 */
public with sharing class CaseLayaRoutingInvocable {

    public class CaseRoutingRequest {
        @InvocableVariable(label='Case Record ID' required=true)
        public Id caseId;

        @InvocableVariable(label='Subject' required=true)
        public String subject;

        @InvocableVariable(label='Description' required=true)
        public String description;

        @InvocableVariable(label='Confidence Gate Threshold' required=false)
        public Decimal confidenceThreshold;
    }

    public class CaseRoutingResult {
        @InvocableVariable(label='Predicted Department')
        public String predictedDepartment;

        @InvocableVariable(label='Confidence Score')
        public Decimal confidence;

        @InvocableVariable(label='Routing Decision (AUTO_ROUTE or HUMAN_TRIAGE_ESCALATION)')
        public String decision;

        @InvocableVariable(label='Routing Reason')
        public String routingReason;

        @InvocableVariable(label='Detected Language')
        public String detectedLanguage;
    }

    @InvocableMethod(label='Route Case with Laya Local Engine' description='Sub-15ms calibrated classification')
    public static List<CaseRoutingResult> routeCases(List<CaseRoutingRequest> requests) {
        List<CaseRoutingResult> results = new List<CaseRoutingResult>();
        List<Case> casesToUpdate = new List<Case>();

        for (CaseRoutingRequest req : requests) {
            HttpRequest httpReq = new HttpRequest();
            // Point to local Laya bridge or secure tunnel / Named Credential
            httpReq.setEndpoint('callout:Laya_Local_Engine/v1/cases/route');
            httpReq.setMethod('POST');
            httpReq.setHeader('Content-Type', 'application/json');

            Map<String, Object> payload = new Map<String, Object>{
                'case_id' => String.valueOf(req.caseId),
                'subject' => req.subject,
                'description' => req.description,
                'confidence_threshold' => (req.confidenceThreshold != null ? req.confidenceThreshold : 0.85)
            };
            httpReq.setBody(JSON.serialize(payload));

            CaseRoutingResult res = new CaseRoutingResult();
            try {
                Http http = new Http();
                HttpResponse httpRes = http.send(httpReq);

                if (httpRes.getStatusCode() == 200) {
                    Map<String, Object> body = (Map<String, Object>) JSON.deserializeUntyped(httpRes.getBody());
                    res.predictedDepartment = (String) body.get('predicted_department');
                    res.confidence = (Decimal) body.get('confidence');
                    res.decision = (String) body.get('decision');
                    res.routingReason = (String) body.get('routing_reason');
                    res.detectedLanguage = (String) body.get('language');

                    // If auto-route threshold is satisfied, assign department directly
                    if (res.decision == 'AUTO_ROUTE') {
                        casesToUpdate.add(new Case(
                            Id = req.caseId,
                            // Replace with your org's Case routing field or Queue Id
                            Type = res.predictedDepartment
                        ));
                    }
                } else {
                    res.decision = 'HUMAN_TRIAGE_ESCALATION';
                    res.routingReason = 'Laya Endpoint Error: HTTP ' + httpRes.getStatusCode();
                }
            } catch (Exception ex) {
                res.decision = 'HUMAN_TRIAGE_ESCALATION';
                res.routingReason = 'Callout exception: ' + ex.getMessage();
            }
            results.add(res);
        }

        if (!casesToUpdate.isEmpty()) {
            update casesToUpdate;
        }

        return results;
    }
}
