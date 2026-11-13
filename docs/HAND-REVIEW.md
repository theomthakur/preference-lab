# Hand review of 40 examples

Method: 25 training rows (from the SFT and preference pools) and 15 validation rows were drawn with a fixed seed (2026). The reviewer read each rendered prompt and the written rules, decided the action and the exact evidence lines, and only then compared with the oracle. Rows were printed without labels.

| # | row | my action | my evidence | oracle action | oracle evidence | agree |
|---|---|---|---|---|---|---|
| 1 | g00243-t1 | DENY | L7 L8 | DENY | L7 L8 | yes |
| 2 | g00654-t0 | HOLD | L1 L9 | HOLD | L1 L9 | yes |
| 3 | g01029-t2 | ACCEPT | L4 L7 L9 | ACCEPT | L4 L7 L9 | yes |
| 4 | g01048-t1 | HOLD | L5 L6 | HOLD | L5 L6 | yes |
| 5 | g01325-t1 | HOLD | L6 L8 L10 | HOLD | L10 L6 L8 | yes |
| 6 | g00210-t1 | HOLD | L1 L4 L5 | HOLD | L1 L4 L5 | yes |
| 7 | g00457-t1 | DENY | L3 L9 | DENY | L3 L9 | yes |
| 8 | g01230-t2 | ACCEPT | L1 L2 L8 | ACCEPT | L1 L2 L8 | yes |
| 9 | g01272-t0 | HOLD | L2 L7 L9 | HOLD | L2 L7 L9 | yes |
| 10 | g01139-t1 | HOLD | L1 L7 | HOLD | L1 L7 | yes |
| 11 | g00861-t2 | DENY | L1 L7 L8 | DENY | L1 L7 L8 | yes |
| 12 | g01172-t0 | HOLD | L4 L10 L11 | HOLD | L10 L11 L4 | yes |
| 13 | g01121-t1 | HOLD | L7 L9 | HOLD | L7 L9 | yes |
| 14 | g01005-t1 | ACCEPT | L1 L2 L9 | ACCEPT | L1 L2 L9 | yes |
| 15 | g01201-t1 | ACCEPT | L4 L8 L10 | ACCEPT | L10 L4 L8 | yes |
| 16 | g00903-t0 | DENY | L3 | DENY | L3 | yes |
| 17 | g00491-t2 | DENY | L7 L10 L11 | DENY | L10 L11 L7 | yes |
| 18 | g00005-t2 | ACCEPT | L5 L6 | ACCEPT | L5 L6 | yes |
| 19 | g01257-t0 | ACCEPT | L1 L3 L7 L10 | ACCEPT | L1 L10 L3 L7 | yes |
| 20 | g00165-t1 | ACCEPT | L3 L7 | ACCEPT | L3 L7 | yes |
| 21 | g00226-t1 | ACCEPT | L1 L7 | ACCEPT | L1 L7 | yes |
| 22 | g00588-t2 | DENY | L3 L5 | DENY | L3 L5 | yes |
| 23 | g00200-t2 | ACCEPT | L1 L6 L9 | ACCEPT | L1 L6 L9 | yes |
| 24 | g00920-t0 | HOLD | L5 L9 | HOLD | L5 L9 | yes |
| 25 | g00023-t0 | ACCEPT | L1 L6 | ACCEPT | L1 L6 | yes |
| 26 | g01575-t2 | HOLD | L5 L11 | HOLD | L11 L5 | yes |
| 27 | g01525-t1 | ACCEPT | L2 L5 | ACCEPT | L2 L5 | yes |
| 28 | g01573-t1 | HOLD | L1 L5 L7 | HOLD | L1 L5 L7 | yes |
| 29 | g01480-t1 | DENY | L2 L8 L9 | DENY | L2 L8 L9 | yes |
| 30 | g01453-t1 | ACCEPT | L7 L8 L10 | ACCEPT | L10 L7 L8 | yes |
| 31 | g01501-t1 | ACCEPT | L2 L8 L10 | ACCEPT | L10 L2 L8 | yes |
| 32 | g01464-t1 | DENY | L1 L4 L9 | DENY | L1 L4 L9 | yes |
| 33 | g01489-t1 | DENY | L2 L7 L10 | DENY | L10 L2 L7 | yes |
| 34 | g01491-t0 | HOLD | L1 L4 | HOLD | L1 L4 | yes |
| 35 | g01496-t1 | DENY | L2 L9 | DENY | L2 L9 | yes |
| 36 | g01590-t1 | HOLD | L5 L7 L9 | HOLD | L5 L7 L9 | yes |
| 37 | g01531-t1 | ACCEPT | L7 L8 | ACCEPT | L7 L8 | yes |
| 38 | g01563-t0 | HOLD | L3 L4 L5 | HOLD | L3 L4 L5 | yes |
| 39 | g01419-t1 | ACCEPT | L10 L11 | ACCEPT | L10 L11 | yes |
| 40 | g01585-t0 | HOLD | L5 L6 L10 | HOLD | L10 L5 L6 | yes |

**Result: 40 of 40 agree on both action and exact evidence.** Labels in the sample: {'DENY': 10, 'HOLD': 15, 'ACCEPT': 15}. Templates: {1: 22, 0: 10, 2: 8}.

What this does and does not show: it checks that the oracle's decisions and evidence ids match a careful human reading of the rendered text for 40 rows, including superseded lines, notes and the rule priority order. It does not prove the rules themselves are a good model of real return policy, and the sample is small. The second-oracle cross-check covers all 1,800 rows separately.
