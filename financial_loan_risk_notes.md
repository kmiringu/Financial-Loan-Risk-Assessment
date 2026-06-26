# Financial Loan Risk — Learning Notes

---

## What the project is asking you to do

You are building a model that helps a loan company decide who to approve for a loan.  
The company currently does this **manually** — a slow, inconsistent process.

Two paths were available:
- **Regression** → predict a *risk score* (a number) that officers use as a guide
- **Classification** → predict *approve or reject* directly (0 or 1)

**We chose classification** because the target column `LoanApproved` is already binary (0 or 1), and the business wants a direct decision, not just a score.

---

## Why these metrics? (Business Understanding)

Not all errors cost the same. That is the key insight here.

| Error Type | What it means | Cost |
|---|---|---|
| **False Positive** | Model says *approve* but the borrower defaults | High — the company loses money |
| **False Negative** | Model says *reject* but the borrower would have repaid | Low — missed revenue only |

Because approving a bad loan is **more costly**, we want a model that is cautious.

### Metrics chosen and why

| Metric | Why we use it |
|---|---|
| **ROC-AUC** | Measures how well the model *ranks* risky vs. safe applicants across all possible thresholds. Works well even when classes are imbalanced (76% rejected vs. 24% approved). A score of 1.0 = perfect; 0.5 = no better than random. |
| **Recall** | Of all the risky borrowers, how many did we catch before approving them? High recall = fewer bad loans slipping through. |
| **F1-Score** | Balances precision and recall. Useful when you need one number to track in production. |

> **Baseline target set:** ROC-AUC ≥ 0.85 (meaning the model must be substantially better than a coin flip).

---

## What the data looks like (Data Understanding)

**Dataset:** `financial_loan_data.csv` — 20,000 loan applications, 34 features, 1 target.

### Feature types — why they matter

| Type | Examples | Challenge |
|---|---|---|
| **Numerical (27)** | CreditScore, LoanAmount, NetWorth, DebtToIncomeRatio | Some are skewed; missing values in SavingsAccountBalance |
| **Ordinal (1)** | EducationLevel | Has a natural order: High School → Doctorate. Cannot treat like a regular category. |
| **Categorical (5)** | EmploymentStatus, MaritalStatus, LoanPurpose | No natural order. Must be one-hot encoded. |
| **Currency string (1)** | AnnualIncome | Stored as `"$39,948.00"` — cannot use as a number until cleaned |

### Missing values

| Feature | Missing | Likely reason |
|---|---|---|
| MaritalStatus | 1,331 (6.7%) | Voluntary non-disclosure |
| EducationLevel | 901 (4.5%) | Not always recorded |
| SavingsAccountBalance | 572 (2.9%) | Applicant may have no savings account |

### Class imbalance

76% of applications were **rejected**, only 24% **approved**.  
This matters because a lazy model that always predicts "rejected" would be 76% accurate — but useless.  
We fix this with `class_weight='balanced'` in the model.

---

## How the data was prepared (Data Preparation)

### The core idea: separate pipelines for separate feature types

Different features need different treatment. We use sklearn's `ColumnTransformer` to route each group through its own mini-pipeline.

```
ColumnTransformer
├── numerical_pipe  → Median imputer → StandardScaler
├── ordinal_pipe    → Mode imputer  → OrdinalEncoder (ranked order)
├── categorical_pipe→ Mode imputer  → OneHotEncoder
└── income_pipe     → IncomeCleaner → Median imputer → StandardScaler
```

### Why each choice was made

**Median imputation (not mean) for numerical features**  
Financial data like income and net worth is skewed by outliers (very wealthy applicants).  
The median is more robust — one billionaire doesn't pull it far from where most people sit.

**OrdinalEncoder for EducationLevel**  
We encode: High School=0, Associate=1, Bachelor=2, Master=3, Doctorate=4.  
This tells the model that Doctorate > Master > Bachelor, which is meaningful.  
Using OneHotEncoder here would lose that ordering.

**OneHotEncoder for other categorical features**  
`LoanPurpose`, `EmploymentStatus` etc. have no natural ranking.  
OrdinalEncoder would imply "Home > Education > Auto" which is meaningless.  
OneHotEncoder creates a separate binary column for each category.

**Custom `IncomeCleaner` transformer**  
`AnnualIncome` arrives as a string: `"$39,948.00"`.  
We write a custom sklearn transformer that strips the `$` and commas so it becomes a usable float.  
It inherits from `BaseEstimator` and `TransformerMixin` so it slots into a pipeline like any other step.

**Why use a Pipeline at all?**  
A pipeline applies the same transformations in order, every time — on training data, on test data, on new data in production.  
Crucially, the imputer and scaler are **fit only on training data** and applied to test data. This prevents **data leakage** (where information from the test set accidentally influences the model).

**Train / Test split with `stratify=y`**  
`stratify=y` ensures both the training set and test set have the same 76/24 class ratio.  
Without it, you could accidentally get a test set that is 90% rejected, which would give misleading evaluation scores.

---

## How the model was built (Modeling)

### Two models compared

**Logistic Regression (baseline)**  
A linear model. It draws a straight line (or hyperplane) through the feature space to separate approved from rejected.  
Fast, interpretable, good for checking whether the problem is linearly separable.

**Random Forest**  
An ensemble of many decision trees. Each tree votes; the majority wins.  
Captures non-linear interactions — e.g., "high debt ratio AND poor payment history" is worse than either alone, and a linear model cannot learn that combination automatically.

Both pipelines look like:
```
Pipeline
├── preprocessor (ColumnTransformer from above)
└── model (LogisticRegression or RandomForestClassifier)
```

Wrapping preprocessing inside the pipeline means cross-validation applies preprocessing correctly — it never "sees" the test fold during fitting.

### Cross-validation: why 5-fold stratified?

Instead of a single train/validation split, we split the training data into 5 parts. Each part takes a turn being the validation set. We average the 5 scores.  
This gives a more reliable estimate of how the model will perform on new data.  
`StratifiedKFold` keeps the 76/24 ratio in each fold.

### GridSearchCV — how hyperparameter tuning works

We give GridSearch a grid of possible settings:

| Parameter | Options | What it controls |
|---|---|---|
| `n_estimators` | 100, 200 | Number of trees. More = lower variance but slower. |
| `max_depth` | 10, 20, None | How deep each tree can grow. Deeper = more complex = risk of overfitting. |
| `min_samples_split` | 2, 5 | Min samples needed to split a node. Higher = simpler trees. |
| `min_samples_leaf` | 1, 2 | Min samples at a leaf. Higher = less sensitive to noise. |

GridSearch tries **every combination** (2 × 3 × 2 × 2 = 24 combinations), evaluating each with 5-fold CV.  
It then selects the combination with the best average ROC-AUC on the validation folds.  
The final model is retrained on the full training set with those best settings.

---

## What the results mean (Evaluation)

### Final test set scores

| Metric | Score |
|---|---|
| ROC-AUC | 0.9994 |
| F1-Score | 0.9780 |
| Recall | 0.9749 |
| Accuracy | 99% |

**ROC-AUC = 0.9994** means: if you pick one random approved applicant and one random rejected applicant, the model ranks the approved one higher 99.94% of the time. Near-perfect discrimination.

**Recall = 0.9749** means: of all the applicants who were actually approved (ground truth), the model correctly identified 97.49% of them.

### The Confusion Matrix explained

```
                 Predicted Rejected  Predicted Approved
Actual Rejected       TN                  FP  ← bad loans approved (costly)
Actual Approved       FN                  TP
```

- **TN (True Negative):** Correctly rejected — good.
- **TP (True Positive):** Correctly approved — good.
- **FP (False Positive):** Approved a bad borrower → **financial loss** — the error we care about most.
- **FN (False Negative):** Rejected a good borrower → lost revenue — less costly.

### The ROC Curve explained

The ROC curve plots **True Positive Rate vs. False Positive Rate** at every possible decision threshold.  
The area under it (AUC) is our primary metric.  
A curve hugging the top-left corner = excellent.  
The dashed diagonal line = a model that is no better than random guessing (AUC = 0.50).

### Feature importance

The Random Forest ranks features by how much each one reduces impurity across all trees.  
Top drivers in this dataset:

1. **RiskScore** — external credit bureau score; the single strongest signal
2. **CreditScore** — borrower's credit history
3. **DebtToIncomeRatio** — total debt relative to income; tests affordability
4. **PaymentHistory** — track record of on-time payments
5. **NetWorth** — overall financial cushion

These match what credit risk theory would predict — the model is learning real signals, not noise.

---

## Key concepts to remember

| Concept | One-line explanation |
|---|---|
| **CRISP-DM** | The 6-step framework: Business → Data → Preparation → Modeling → Evaluation → Deployment |
| **Data leakage** | When test data information accidentally influences training. Pipelines prevent this. |
| **Class imbalance** | When one class is much more common. Fix with `class_weight='balanced'` or resampling. |
| **Stratified split** | Keeps class ratios equal across train/test/CV folds. |
| **ColumnTransformer** | Routes different feature groups to different preprocessing pipelines. |
| **GridSearchCV** | Exhaustive search over a hyperparameter grid, scored by cross-validation. |
| **ROC-AUC** | How well the model ranks examples, regardless of the decision threshold. |
| **Recall** | Of all actual positives, how many did the model catch? |
| **F1-Score** | Harmonic mean of precision and recall. Good single metric when classes are imbalanced. |
| **OrdinalEncoder** | Encodes categories that have a natural order (e.g. education level). |
| **OneHotEncoder** | Encodes categories with no order — creates one binary column per category. |
| **Custom Transformer** | A class inheriting `BaseEstimator` + `TransformerMixin` that plugs into any sklearn pipeline. |

---

## Things to watch out for in real deployments

**RiskScore as a feature**  
It is the top predictor. Before deploying, confirm it is an *external* bureau score available at application time — not a score derived *from* the loan decision. If it is post-decision, including it is data leakage and the model would fail completely in production.

**High scores ≠ real-world guarantee**  
ROC-AUC of 0.9994 is unusually high, which suggests the dataset may be synthetic or pre-cleaned. In real lending data, expect ROC-AUC in the 0.78–0.88 range. The methodology and pipeline are correct regardless.

**Fairness**  
A model trained on historical decisions inherits any bias those decisions contained. Before deployment, check approval rates across age, marital status, and employment type to satisfy fair lending regulations.

**Threshold is a business decision**  
The model outputs a probability (0–1). The default decision boundary is 0.50.  
Raising it to 0.65 means fewer approvals but fewer defaults — a risk-appetite choice for management, not a modeling choice.
