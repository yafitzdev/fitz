<!-- benchmarks/corpora/holdout2/unstructured/incident_review.md -->
# Incident Review Log

## INC-611 Payment Queue Stall

Initial update on 2026-04-03 estimated 18 minutes of customer-visible impact.

Final review on 2026-04-04 confirmed 42 minutes of customer-visible outage.
Commander was Dana Ortiz. Root cause was the replay worker starving the
idempotency queue.

## INC-724 Search Index Lag

Final recovery duration was 26 minutes. Commander was Rowan Vale. The incident
was not customer visible.

## SEC-144 Token Exposure

Security notification for SEC-144 was sent within 24 hours by CISO Arun Mehta.
