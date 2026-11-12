# Final agent experiment audit

## Aider

- Attempted/valid: 225/174
- Draft/final pass rate: 0.1289 / 0.3422
- Format compliance: 0.4988
- Paired attribution intersection: 174
- IFR last: RISE 0.1863, MAS 0.2791, time 4.09s
- IFR exhaustive: RISE 0.1149, MAS 0.2346, time 368.65s
- AttnLRP: RISE 0.2063, MAS 0.3332, time 3.72s
- FlashTrace: RISE 0.1206, MAS 0.2560, time 6.25s

## AgentDojo

- Clean/attacked inventory: 97/949
- Clean/attacked utility: 0.6701 / 0.6417
- Attack success: 0.0537
- Localizable: 30/949 (0.0316)
- Successful attribution: 30/30
- Random: R@5 0.0502, R@20 0.2002, time 0.00s
- IFR last: R@5 0.0510, R@20 0.3973, time 1.68s
- IFR full: R@5 0.1408, R@20 0.5972, time 18.00s
- FlashTrace: R@5 0.1509, R@20 0.6316, time 2.35s

All confidence intervals use 50,000 paired sample-bootstrap draws.
