<!-- benchmarks/corpora/holdout/unstructured/payments_postmortem.md -->
# Payments Postmortems

## Initial PAY-209 Status

The first status update estimated that incident PAY-209 would recover in
12 minutes. That estimate was issued before the webhook deadlock was isolated.

## Final PAY-209 Postmortem

The final postmortem confirmed that incident PAY-209 recovered after
37 minutes. Imani owned the incident. The root cause was an invoice webhook
deadlock, and the alert route was ALT-501.

## Privacy Incident PRI-88

DPO Lena owned privacy incident PRI-88. The incident involved customer export
logs, resolved on 2026-05-09, and carried a 72-hour notification window.
