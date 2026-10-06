# Dataset provenance and dictionary

The dataset is **UCI Bank Marketing, bank-additional-full**, by Sergio Moro, Paulo Cortez, and Paulo Rita. It describes a Portuguese bank's term deposit telemarketing contacts, ordered from May 2008 to November 2010. One row is a contact observation, not a verified unique customer.

## Files

- `raw/bank-additional-full.csv`: the unchanged official semicolon-delimited CSV, 41,188 observations and 21 columns.
- `raw/bank-additional-names.txt`: the unchanged official English dictionary and research attribution.
- `manifests/dataset_manifest.json`: original source, variant, checksums, and ingestion provenance.
- `manifests/audit.json`: recorded audit, errors/warnings, category counts, and descriptive statistics.
- `manifests/split_manifest.json`: fixed original-position split boundaries, prevalences, and duplicate overlap.

CSV SHA-256: `74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8`.

Dictionary SHA-256: `10045462a4c026f858dffa159370c4270510230c1fc63c6c91ca51bba4b3df26`.

The 4,640 yes and 36,548 no outcomes correspond to 11.2654% yes. Audit identifies 12 exact duplicated rows, 0 errors, and 4,110 warnings. The app preserves raw data and reports warnings; it does not deduplicate or silently impute the official file. Unknown values include default 8,597; education 1,731; housing 990; loan 990; job 330; marital 80.

## Original fields

| Field | Meaning | Main model |
| --- | --- | --- |
| `age` | Age in years | Included |
| `job` | Employment category | Included |
| `marital` | Marital status; divorced also includes widowed | Included |
| `education` | Education category | Included |
| `default` | Credit in default | Included |
| `housing` | Housing loan | Included |
| `loan` | Personal loan | Included |
| `contact` | Current contact communication type | Excluded |
| `month` | Last current-campaign contact month | Excluded |
| `day_of_week` | Last current-campaign contact weekday | Excluded |
| `duration` | Last contact duration in seconds, unavailable before that call | Excluded |
| `campaign` | Current-campaign contact count, including last contact | Excluded |
| `pdays` | Days since previous campaign contact; 999 means never contacted | Included with sentinel transformation |
| `previous` | Contacts before current campaign | Included |
| `poutcome` | Previous campaign outcome | Included |
| `emp.var.rate` | Quarterly employment variation rate | Excluded |
| `cons.price.idx` | Monthly consumer price index | Excluded |
| `cons.conf.idx` | Monthly consumer confidence index | Excluded |
| `euribor3m` | Daily 3-month Euribor rate | Excluded |
| `nr.employed` | Quarterly employee count indicator | Excluded |
| `y` | Term deposit subscription: yes/no | Target, never a feature |

The official dictionary defines original meanings; `docs/REQUIREMENTS.md` defines the stricter application input schema. `unknown` is a category, not a license to replace malformed input. Derived IDs are internal row positions, not public customer identifiers.

## Attribution and license

Dataset landing page: https://archive.ics.uci.edu/dataset/222/bank+marketing

Official download: https://archive.ics.uci.edu/static/public/222/bank%2Bmarketing.zip

License: **Creative Commons Attribution 4.0 International**, https://creativecommons.org/licenses/by/4.0/ . The CSV and dictionary are redistributed unchanged. Models, manifests, charts, and application code are derived project artifacts; the code's MIT license does not replace the dataset license.

Moro, S., Cortez, P., and Rita, P. (2014). *A Data-Driven Approach to Predict the Success of Bank Telemarketing*. Decision Support Systems, 62, 22-31. https://doi.org/10.1016/j.dss.2014.03.001
