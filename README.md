# Mall-Customer-Segmentation-KMeans

Customer segmentation on mall shopper data using K-Means clustering, with marketing strategy recommendations for each discovered segment.

## Task Objective

Segment mall customers into distinct groups based on their annual income and spending behavior, so that targeted marketing strategies can be designed for each group rather than applying a one-size-fits-all approach.

## Approach

1. **Data loading** — loaded `Mall_Customers.csv` into a pandas DataFrame.
2. **Exploratory Data Analysis (EDA)** — checked data types and missing values, reviewed descriptive statistics, and visualized the distributions of age, annual income, and spending score, along with the gender split.
3. **Feature scaling** — selected Annual Income and Spending Score as clustering features and standardized them with `StandardScaler`, since K-Means relies on distance and is sensitive to feature scale.
4. **Determining cluster count** — applied the Elbow Method (plotting within-cluster sum of squares for k = 1 to 10) to choose an appropriate number of clusters.
5. **K-Means clustering** — fit K-Means with 5 clusters on the scaled features and assigned each customer a cluster label.
6. **Cluster profiling** — computed the average age, income, and spending score, plus the most common gender, for each cluster to understand its characteristics.
7. **Marketing strategy mapping** — translated each cluster's profile into a tailored marketing recommendation.
8. **t-SNE visualization** — used t-SNE to project the scaled features into 2D as an additional check on how well-separated the clusters are.

## Results and Findings

K-Means with 5 clusters produced clearly separated, interpretable segments based on income and spending score:

- **High Income, Low Spending — "Careful Spenders":** affluent but budget-conscious; respond best to value-for-money offers, loyalty programs, and premium long-term products rather than impulse-buy promotions.
- **Low Income, High Spending — "Impulse Buyers":** spend heavily relative to income, likely trend-driven; respond well to discounts, payment plans, flash sales, and social-media-led campaigns.
- **High Income, High Spending — "Target Customers":** the most valuable segment; warrant VIP treatment, exclusive previews, and premium, personalized service.
- **Low Income, Low Spending — "Budget-Conscious":** price-sensitive and focused on essentials; best served with budget-friendly products, bulk discounts, and practical value messaging.
- **Mid Income, Mid Spending — "Mainstream":** the largest, most average segment; suited to broad promotions, standard loyalty programs, and general advertising.

The Elbow Method confirmed 5 as a reasonable choice of k, and the t-SNE plot showed the clusters remained visually distinct when projected to 2D, supporting the validity of the K-Means segmentation. Overall, income and spending score alone were sufficient to produce actionable, business-relevant customer segments.

## Dataset

| Column | Description |
|---|---|
| CustomerID | Unique identifier for each customer |
| Genre | Gender of the customer |
| Age | Customer's age |
| Annual Income (k$) | Annual income in thousands of dollars |
| Spending Score (1-100) | Score assigned by the mall based on spending behavior |

This is the standard "Mall Customer Segmentation Data" dataset (commonly found on Kaggle).

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

## Usage

1. Place `Mall_Customers.csv` in your working directory (the notebook's Google Drive/zip-extraction cells are only needed if running on Colab with the file stored that way; locally, load the CSV directly with `pandas.read_csv`).
2. Run the notebook cells in order — EDA and visualizations, the elbow plot, K-Means fitting and cluster profiling, marketing recommendations, and the t-SNE plot.

## Notes

- `random_state=42` is used throughout for reproducibility.
- The Colab-specific cells (`drive.mount`, zip extraction) can be removed or replaced with a direct CSV path when running outside of Google Colab.
