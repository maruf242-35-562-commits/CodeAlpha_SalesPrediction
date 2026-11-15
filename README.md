# CodeAlpha_SalesPrediction

**CodeAlpha Data Science Internship – Task 4: Sales Prediction using Python**

Predicts product sales from advertising spend on TV, Radio and Newspaper (200 markets) and analyses how changes in each channel affect sales.

## Files
| File | Purpose |
|---|---|
| `Sales_Prediction.ipynb` | Full workflow: cleaning, EDA, feature selection, modelling, diagnostics, impact and budget analysis |
| `sales_model.joblib` | Trained final model |
| `data/Advertising.csv` | Dataset |
| `images/` | Charts exported from the notebook |

## Approach
1. **Cleaning:** dropped the leftover index column; no missing values or duplicates.
2. **Feature selection:** Newspaper looks useful alone but has ~0 effect once TV and Radio are included (p = 0.86), no CV gain, and LASSO sets it to 0, so it is removed.
3. **Feature engineering:** TV shows diminishing returns and TV x Radio shows synergy, so polynomial terms are used.
4. **Models compared** with 5-fold x 10-repeat CV and an 80/20 hold-out: linear baselines, interaction model, degree-2 and degree-3 (Ridge) polynomials, Random Forest, Gradient Boosting.
5. **Impact analysis:** marginal effects per extra $1k, response surface, what-if scenarios, budget allocation, bootstrap uncertainty.

## Results
| Model | CV R² | CV RMSE (k units) |
|---|---|---|
| Linear baseline (TV, Radio, Newspaper) | 0.884 | 1.70 |
| + TV x Radio interaction | 0.965 | 0.94 |
| Degree-2 polynomial | 0.985 | 0.61 |
| **Degree-3 polynomial + Ridge (final)** | **0.990** | **0.50** |
| Random Forest | 0.976 | 0.78 |
| Gradient Boosting | 0.979 | 0.73 |

Hold-out R² ≈ 0.994.

## Key insights
- **Newspaper adds nothing** once TV and Radio are known.
- **TV has diminishing returns**, and **TV and Radio reinforce each other**.
- At an average market, an extra $1k on **radio adds ~0.20k units vs ~0.03k for TV** (about 6x), and the gap holds in 100% of bootstrap refits.
- At a $150k TV+Radio budget, putting radio at its observed maximum (~$49.5k) and the rest in TV is predicted to sell ~25% more than the typical split. The optimum sits at the edge of the data, so do not extrapolate beyond it, and treat results as associations that should be pilot-tested.


## Predict sales
```python
import joblib, pandas as pd
model = joblib.load("sales_model.joblib")
print(model.predict(pd.DataFrame({"TV": [150], "Radio": [20]})))   # thousand units
```


