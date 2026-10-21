/**
 * @description Example Apex Trigger demonstrating asynchronous dispatch to Laya
 * for newly inserted or updated Cases.
 */
trigger CaseRoutingTrigger on Case (after insert) {
    List<CaseLayaRoutingInvocable.CaseRoutingRequest> routingRequests = new List<CaseLayaRoutingInvocable.CaseRoutingRequest>();

    for (Case c : Trigger.new) {
        if (c.Type == null && c.Subject != null) {
            CaseLayaRoutingInvocable.CaseRoutingRequest req = new CaseLayaRoutingInvocable.CaseRoutingRequest();
            req.caseId = c.Id;
            req.subject = c.Subject;
            req.description = (c.Description != null ? c.Description : c.Subject);
            req.confidenceThreshold = 0.85;
            routingRequests.add(req);
        }
    }

    if (!routingRequests.isEmpty() && !System.isFuture() && !System.isBatch()) {
        // Enqueue or execute async callout to preserve governor limits
        CaseLayaRoutingInvocable.routeCases(routingRequests);
    }
}
