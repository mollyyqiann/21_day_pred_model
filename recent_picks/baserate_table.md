# Monthly Gainer Base Rate Study

**Overall**: 366,555 labeled rows, 4,699 positives, base rate **1.2819%**

## By year

| year | n | pos | rate |
|---|---:|---:|---:|
| 2023 | 84,220 | 835 | 0.991% |
| 2024 | 126,135 | 1,280 | 1.015% |
| 2025 | 125,517 | 1,866 | 1.487% |
| 2026 | 30,683 | 718 | 2.340% |

## By GICS sector

| sector | n | pos | rate | top tickers |
|---|---:|---:|---:|---|
| Information Technology | 52,362 | 2,109 | 4.028% | APP(200), SMCI(165), LITE(143) |
| Consumer Discretionary | 35,136 | 651 | 1.853% | CVNA(219), TSLA(102), NCLH(54) |
| Communication Services | 16,836 | 268 | 1.592% | SATS(111), PSKY(48), WBD(43) |
| Utilities | 22,692 | 218 | 0.961% | VST(96), CEG(79), NRG(25) |
| Financials | 55,631 | 469 | 0.843% | COIN(173), HOOD(163), XYZ(47) |
| Industrials | 57,492 | 478 | 0.831% | VRT(119), FIX(65), GEV(50) |
| Materials | 19,032 | 119 | 0.625% | ALB(43), CF(15), LYB(15) |
| Energy | 16,104 | 98 | 0.609% | TPL(62), APA(20), MPC(6) |
| Health Care | 42,229 | 232 | 0.549% | MRNA(87), DVA(27), PODD(16) |
| Consumer Staples | 26,349 | 37 | 0.140% | DG(12), EL(10), BF-B(8) |
| Real Estate | 22,692 | 20 | 0.088% | BXP(9), SBAC(7), EXR(2) |

## By VIX regime (terciles)

| regime | vix range | n | pos | rate |
|---|---|---:|---:|---:|
| low | 11.9-14.8 | 121,932 | 1,067 | 0.875% |
| mid | 14.8-17.5 | 122,259 | 1,372 | 1.122% |
| high | 17.5-52.3 | 122,364 | 2,260 | 1.847% |

## By rv_60 quartile (stock-level vol)

| bucket | rv range | n | pos | rate |
|---|---|---:|---:|---:|
| Q1 | 0.03-0.21 | 84,094 | 36 | 0.043% |
| Q2 | 0.21-0.26 | 84,093 | 118 | 0.140% |
| Q3 | 0.26-0.34 | 84,094 | 449 | 0.534% |
| Q4 | 0.34-2.08 | 84,094 | 3,657 | 4.349% |
| unknown | nan-nan | 30,180 | 439 | 1.455% |

## By run_length bucket (already-trending state)

| bucket | n | pos | rate |
|---|---:|---:|---:|
| 0-5 | 227,673 | 3,020 | 1.326% |
| 5-20 | 95,664 | 1,050 | 1.098% |
| 20-60 | 41,833 | 566 | 1.353% |
| 60+ | 1,385 | 63 | 4.549% |

## Concentration

- 176 of 503 tickers ever fired (≥1 positive)
- top-10 tickers contribute **33.1%** of all positives
- top-25 tickers contribute **59.0%** of all positives
- top-50 tickers contribute **80.5%** of all positives
- top-100 tickers contribute **94.1%** of all positives

**Top 15 tickers by positive count:**
| ticker | positives |
|---|---:|
| CVNA | 219 |
| APP | 200 |
| COIN | 173 |
| SMCI | 165 |
| HOOD | 163 |
| LITE | 143 |
| SNDK | 135 |
| PLTR | 119 |
| VRT | 119 |
| COHR | 118 |
| SATS | 111 |
| TSLA | 102 |
| MU | 97 |
| VST | 96 |
| STX | 88 |

## Event clustering

- Fresh events (no positive in prior 7 days): 585 (12.4%)
- Continuation events: 4,114