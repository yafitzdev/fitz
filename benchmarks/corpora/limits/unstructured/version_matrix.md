# Version Matrix

## RX Monitor Historical

RX Monitor retained sensor frames for 14 days in release REL-2025.11. That historical value is audit-only.

## RX Monitor Current

RX Monitor retains sensor frames for 21 days in release REL-2026.03. This current value is used in production.

## Queue Flush Historical

Queue Flush used retry backoff 400 ms in release REL-2025.12. That value is deprecated.

## Queue Flush Current

Queue Flush uses retry backoff 650 ms in release REL-2026.04. This current value is used for live traffic.

## Payment Sync Draft

Payment Sync draft notes estimated retry budget 2 attempts for package PAY-7.

## Payment Sync Final

Payment Sync final notes confirmed retry budget 5 attempts for package PAY-7.

## Gamma Router Legacy

Gamma Router had route limit 70 hops in release REL-2024.09. The legacy value is retained only for trace replay.

## Gamma Router Current

Gamma Router has route limit 48 hops in release REL-2026.01. The current value applies to all production routes.
