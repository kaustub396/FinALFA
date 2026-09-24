import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef, balanced_accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('data/final_df.csv')
y = df['market_up_next'].values
F = df['Fundamental_Score_FF'].values
S = df['Annual_Sentiment_Norm'].values
returns = df['annual_return'].values

N = len(y)
alphas = np.linspace(0, 1, 21)

def run_finalfa_loocv(y_true, F_data, S_data):
    preds = np.zeros(N)
    best_alphas = np.zeros(N)
    
    for i in range(N):
        train_idx = [j for j in range(N) if j != i]
        
        y_train = y_true[train_idx]
        F_train = F_data[train_idx]
        S_train = S_data[train_idx]
        
        best_alpha = 0.5
        best_inner_acc = -1
        
        for alpha in alphas:
            x_alpha_train = (alpha * S_train + (1 - alpha) * F_train).reshape(-1, 1)
            inner_preds = np.zeros(len(y_train))
            for k in range(len(y_train)):
                inner_train_idx = [j for j in range(len(y_train)) if j != k]
                
                clf = LogisticRegression(penalty='l2', C=1.0)
                clf.fit(x_alpha_train[inner_train_idx], y_train[inner_train_idx])
                inner_preds[k] = clf.predict(x_alpha_train[k].reshape(1, -1))[0]
                
            inner_acc = accuracy_score(y_train, inner_preds)
            if inner_acc > best_inner_acc:
                best_inner_acc = inner_acc
                best_alpha = alpha
                
        best_alphas[i] = best_alpha
        
        # Train final model for fold i
        x_alpha_train = (best_alpha * S_train + (1 - best_alpha) * F_train).reshape(-1, 1)
        clf = LogisticRegression(penalty='l2', C=1.0)
        clf.fit(x_alpha_train, y_train)
        
        x_alpha_test = (best_alpha * S_data[i] + (1 - best_alpha) * F_data[i]).reshape(1, -1)
        preds[i] = clf.predict(x_alpha_test)[0]
        
    return preds, best_alphas

def run_gb_loocv(y_true, X_data):
    preds = np.zeros(N)
    for i in range(N):
        train_idx = [j for j in range(N) if j != i]
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        clf.fit(X_data[train_idx], y_true[train_idx])
        preds[i] = clf.predict(X_data[i].reshape(1, -1))[0]
    return preds

print("Running baseline FinALFA...")
finalfa_preds, finalfa_alphas = run_finalfa_loocv(y, F, S)

X_gb = (F * S).reshape(-1, 1)
print("Running Gradient Boosting...")
gb_preds = run_gb_loocv(y, X_gb)

def print_metrics(name, y_true, y_pred):
    print(f"--- {name} ---")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_true, y_pred):.3f}")
    print(f"MCC: {matthews_corrcoef(y_true, y_pred):.3f}")
    print(f"Precision (Class 0): {precision_score(y_true, y_pred, pos_label=0, zero_division=0):.3f}")
    print(f"Recall (Class 0): {recall_score(y_true, y_pred, pos_label=0, zero_division=0):.3f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}")
    print()

print_metrics("Majority Baseline", y, np.ones(N))
print_metrics("Gradient Boosting", y, gb_preds)
print_metrics("FinALFA", y, finalfa_preds)

print("--- FinALFA Fold Details ---")
df_folds = pd.DataFrame({
    'Year': df['Year'],
    'Actual': y,
    'Predicted': finalfa_preds,
    'Alpha_Star': finalfa_alphas
})
print(df_folds)
print()

# Permutation Test
print("Running Permutation Test (n=100)...")
import concurrent.futures

def single_permutation(seed, y_true, F_data, S_data):
    np.random.seed(seed)
    y_perm = np.random.permutation(y_true)
    p_preds, _ = run_finalfa_loocv(y_perm, F_data, S_data)
    return accuracy_score(y_perm, p_preds)

if __name__ == '__main__':
    n_permutations = 100
    perm_accs = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(single_permutation, 42+p, y, F, S) for p in range(n_permutations)]
        for i, f in enumerate(concurrent.futures.as_completed(futures)):
            perm_accs.append(f.result())
            if (i+1) % 10 == 0:
                print(f"  Completed {i+1}/{n_permutations}")

    perm_accs = np.array(perm_accs)
    actual_acc = accuracy_score(y, finalfa_preds)
    p_val = np.mean(perm_accs >= actual_acc)
    print(f"Permutation p-value: {p_val:.3f}")
    print(f"Permutation mean acc: {np.mean(perm_accs):.3f}, std: {np.std(perm_accs):.3f}")

    # Portfolio Simulation
    print("\n--- Portfolio Simulation ---")
    def sim_portfolio(preds, rets, tc=0.001):
        # Start with 1.0
        value = 1.0
        values = [value]
        prev_pos = 1 # assume fully invested initially
        
        for i in range(len(preds)):
            target_pos = 1 if preds[i] == 1 else 0 # Long or Cash
            
            # Trade cost if position changes
            if target_pos != prev_pos:
                value *= (1 - tc)
                
            # Return
            if target_pos == 1:
                value *= (1 + rets[i])
                
            values.append(value)
            prev_pos = target_pos
        return values[-1]

    val_bh = sim_portfolio(np.ones(N), returns, tc=0.0)
    val_finalfa_0 = sim_portfolio(finalfa_preds, returns, tc=0.0)
    val_finalfa_tc = sim_portfolio(finalfa_preds, returns, tc=0.001)

    print(f"Buy and Hold (0% TC): {val_bh:.3f}")
    print(f"FinALFA (0% TC): {val_finalfa_0:.3f}")
    print(f"FinALFA (0.1% TC): {val_finalfa_tc:.3f}")


