| Metric | Value |
|---|---|
| Songs indexed | 10,000 |
| Features | duration, tempo, loudness, key, mode, time_signature |
| Spark master / parallelism | local[*] / 24 |
| LSH params | bucketLength=0.5, numHashTables=5 |
| Parquet load time | 2.125 s |
| Fit time (impute + scale + normalize + LSH, cached) | 5.061 s |
| LSH query latency, mean / median / p95 (20 queries) | 0.299 / 0.273 / 0.412 s |
| Exact (full scan) query latency, mean | 0.217 s |
| Recall@10 of LSH vs exact, mean (min) | 1.0 (1.0) |
| Mean cosine similarity of recommendations | 0.9808 |

**Sample output** - seed: *Diamond In The Rough* by Syl Johnson

| # | Title | Artist | Cosine similarity |
|---|---|---|---|
| 1 | Acordeón | Massiel | 0.9886 |
| 2 | More Than One Love | Venice | 0.9884 |
| 3 | Stickin In My Eye | NOFX | 0.9867 |
| 4 | Tu Peux Compter Sur Moi | Bénabar | 0.9832 |
| 5 | Grin'in In Ya Face | Smidi Beats | 0.9824 |
| 6 | Shame | Anna Abreu | 0.9784 |
| 7 | Fotia Sta Savvatovrada (Live) | Angela Dimitriou | 0.9783 |
| 8 | Panda Bear | Owl City | 0.9781 |
| 9 | On and on | Franco | 0.9751 |
| 10 | Lil Mama Getdown | DJ Spinn featuring Traxman | 0.9715 |
