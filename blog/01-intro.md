# Rules Disguised as AI: The $50M Support Routing Illusion

*By Pavan Chavali · 7 min read*

---

If you have spent five minutes in an enterprise IT architecture review, you have heard someone declare: *"We have AI routing our customer support tickets."*

Nine times out of ten, if you open Salesforce Setup, navigate to **Case Assignment Rules**, and examine what is actually running, you find this:

```
IF Subject CONTAINS "invoice" OR "bill" THEN RouteTo: Billing_Queue
IF Body CONTAINS "error 500" OR "API" THEN RouteTo: Engineering_Tier2
IF Subject CONTAINS "discount" OR "renewal" THEN RouteTo: Inside_Sales
```

This is not artificial intelligence. It is a fragile house of cards built on string matching and boolean conjunctions.

Yet across Global 2000 enterprises, thousands of engineering hours are consumed debugging these rule sets. A rule added in March to catch billing disputes accidentally intercepts enterprise renewal contracts in August. A support case written in conversational Spanish or Hindi bypasses every rule entirely and lands in the "General Unassigned" black hole.

And most dangerously: **Rules have zero concept of uncertainty.**

A regex rule cannot tell you, *"I matched this case with 98% confidence,"* or *"I matched this case, but there is a 60% chance it is actually an account cancellation."* Every case is auto-routed with 100% blind conviction. The error rate is paid for in customer frustration, missed SLAs, and endless internal ticket ping-pong.

In this series, we unpack why the industry’s two traditional answers—**handcrafted rules** and **closed SaaS AI (like Salesforce Einstein)**—both fail modern architectural standards, and why **local-first, composable decision engines like Laya** represent the future of intelligent workflow routing.

Next up in Part 2: *Why Closed AI Like Salesforce Einstein Traps Your Enterprise.*
