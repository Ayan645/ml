# ML Programs — Linux Execution

Run these commands from the extracted project root. Set up the shared Python environment first:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

| Program | Dataset | Command |
|---|---|---|
| Gradient Descent | None | `python3 ml/Gradient_Descent_Algorithm.py` |
| KNN Diabetes | `diabetes.csv` | `./download_datasets.sh diabetes` then `python3 ml/K_Nearest_Neighbors.py` |
| K-Means Sales | `sales_data_sample.csv` | `./download_datasets.sh sales` then `python3 ml/k_means_hierarchical_clustering.py` |
| Uber Fare Prediction | `uber.csv` | `./download_datasets.sh uber` then `python3 ml/Uber_ride.py` |
| Email Spam Detection | `emails.csv` | `./download_datasets.sh emails` then `python3 ml/Email_Spam.py` |

The emails and Uber datasets are large. Only download the dataset needed for the experiment. Run the download script from the project root. The K-Means file is named after the original repository file but its code implements K-Means, not hierarchical clustering.
