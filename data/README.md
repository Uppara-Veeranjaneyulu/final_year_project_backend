# Workload Datasets Directory

This directory manages the 10 real-world benchmark workload datasets evaluated in the research paper:

1. **Google Cluster Workload Trace v1 (2011)** — [GitHub](https://github.com/google/cluster-data)
2. **Google Cluster Trace v2.1 (2019)** — [Google Storage](https://storage.googleapis.com/clusterdata-2011-2/)
3. **Bitbrains Cloud Workload (GWA-T-12)** — [GWA Archive](https://atlarge-research.com/gwa-t-12/)
4. **Azure Public Dataset (2017)** — [GitHub](https://github.com/Azure/AzurePublicDataset)
5. **Alibaba Cluster Trace (2018)** — [GitHub](https://github.com/alibaba/clusterdata)
6. **Spitzer Space Telescope Logs** — [NASA IRSA](https://irsa.ipac.caltech.edu/data/SPITZER/docs/files/spitzer/)
7. **XMM-Newton Observation Logs** — [ESA XSA](https://nxsa.esac.esa.int/ftp_public/heasarc_obslog/)
8. **Parallel Workloads Archive (PWA)** — [HUJI PWA](https://www.cs.huji.ac.il/labs/parallel/workload/logs.html)
9. **HPC2N Workload Dataset** — [HPC2N Log](https://www.cs.huji.ac.il/labs/parallel/workload/l_hpc2n/)
10. **CAIDA Internet Traffic Dataset (2025)** — [CAIDA Catalog](https://catalog.caida.org/dataset/passive_2025_pcap_100g)

---

## Directory Organization

```
final_year_project_backend/data/
├── raw/                 # Downloaded raw CSV / TXT / PCAP logs
├── processed/           # 5-minute resampled & MinMax normalized numpy/pandas tensors
├── download_datasets.py # Automated fetcher script for public trace data
└── dataset_loader.py    # Python data loader module for SARIMAX & Gymnasium env
```

---

## Quick Start: Download & Generate Traces

Run the dataset download & processing helper:
```powershell
..\.venv\Scripts\python download_datasets.py
```
