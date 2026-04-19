# Data Cleaning and EDA Summary

## Proposal Alignment

- Raw OHLCV timestamps are standardized to UTC-style ISO output files.
- Stock files now include technical indicators: moving averages, EMA, RSI, volatility, and next-bar direction labels.
- News articles are cleaned and deduplicated before being written to derived outputs.
- Relevant AAPL and AMZN articles are aligned to the next valid market bar for each stock interval.
- Chronological 80/20 split metadata is exported to support leakage-safe modeling.

## Stock Dataset Summary

- Tickers found: AAPL, AMZN
- Intervals found (minutes): 5, 15, 30, 60, 240, 1440
- Source stock files: 12
- Total stock rows processed: 133295
- Invalid or skipped stock rows: 0
- Duplicate stock timestamps detected: 0

| Ticker | Interval | Rows | Date Min | Date Max | Mean Close | Avg Volume | Volatility % |
|---|---:|---:|---|---|---:|---:|---:|
| AAPL | 1440 | 1143 | 2014-06-30 | 2019-02-01 | 137.452 | 48083.6 | 1.5274 |
| AAPL | 15 | 17495 | 2016-12-22 | 2019-02-01 | 168.637 | 1723.7 | 0.2540 |
| AAPL | 240 | 3246 | 2014-06-30 | 2019-02-01 | 137.215 | 16931.5 | 0.8885 |
| AAPL | 30 | 10178 | 2016-08-23 | 2019-02-01 | 160.699 | 3370.1 | 0.3480 |
| AAPL | 5 | 38634 | 2017-07-12 | 2019-02-01 | 179.061 | 607.5 | 0.1632 |
| AAPL | 60 | 6393 | 2016-03-16 | 2019-02-01 | 151.386 | 6163.2 | 0.4788 |
| AMZN | 1440 | 1094 | 2014-09-29 | 2019-02-01 | 921.585 | 20422.5 | 1.8766 |
| AMZN | 15 | 11234 | 2017-09-25 | 2019-02-01 | 1532.783 | 916.1 | 0.3581 |
| AMZN | 240 | 2266 | 2015-11-27 | 2019-02-01 | 1097.190 | 8002.9 | 1.0791 |
| AMZN | 30 | 7155 | 2017-05-12 | 2019-02-01 | 1415.262 | 1684.0 | 0.4587 |
| AMZN | 5 | 30283 | 2017-11-09 | 2019-02-01 | 1589.798 | 318.3 | 0.2207 |
| AMZN | 60 | 4174 | 2017-03-13 | 2019-02-01 | 1367.754 | 3069.3 | 0.6217 |

## News Dataset Summary

- Raw articles scanned: 78055
- Deduplicated articles retained: 77766
- Missing published timestamps: 0
- Missing text bodies: 32
- Duplicate UUIDs removed: 287
- Duplicate fingerprint matches removed: 2
- Published UTC range: 2017-12-07T20:00:00+00:00 to 2019-02-07T23:10:00+00:00
- Average text length: 4958.7 characters
- Median text length: 4271.0 characters
- Average word count: 784.5
- Median word count: 695.0
- Relevance-filtered ticker mentions: AAPL=77750, AMZN=21232
- Aligned market bars with at least one article: 45562

### Top News Sites

| Site | Articles |
|---|---:|
| yahoo.com | 6249 |
| morningstar.com | 5146 |
| marketwatch.com | 4616 |
| seekingalpha.com | 3852 |
| nasdaq.com | 3838 |
| zacks.com | 2004 |
| mmahotstuff.com | 1969 |
| fool.com | 1958 |
| thestreet.com | 1721 |
| investing.com | 1381 |

### Top Languages

| Language | Articles |
|---|---:|
| english | 70451 |
| chinese | 1541 |
| chineset | 1326 |
| japanese | 1133 |
| german | 770 |
| russian | 581 |
| french | 347 |
| spanish | 318 |
| italian | 241 |
| polish | 188 |

### Top Countries

| Country | Articles |
|---|---:|
| US | 64468 |
| CN | 1341 |
| GB | 1240 |
| JP | 1192 |
| DE | 1133 |
| IN | 826 |
| EU | 795 |
| TW | 754 |
| CA | 612 |
| FR | 607 |

### Top Organization Mentions

| Organization | Mentions |
|---|---:|
| apple | 32947 |
| aapl | 19761 |
| s&p | 18071 |
| apple inc. | 10571 |
| sec | 9715 |
| nyse | 7338 |
| apple inc | 6329 |
| reuters | 4748 |
| facebook | 4031 |
| google | 3854 |
| amazon | 3319 |
| amzn | 2950 |
| microsoft | 2869 |
| netflix | 2756 |
| nflx | 2270 |

### Top Person Mentions

| Person | Mentions |
|---|---:|
| trump | 3143 |
| donald trump | 3020 |
| tim cook | 2526 |
| marketwatch | 1946 |
| warren buffett | 1457 |
| tesla | 1212 |
| buffett | 1012 |
| cook | 906 |
| mark decambre | 893 |
| barbara kollmeyer | 695 |
| jim cramer | 663 |
| spotify | 627 |
| ryan vlastelica | 567 |
| jerome powell | 567 |
| sue chang | 519 |
