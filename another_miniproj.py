import numpy as np
import matplotlib.pyplot as plt

# Step 1
np.random.seed(1)

w_true = np.array([0, 2.5, -0.5])
sigma = 0.3

lin_size = 41
x1 = np.linspace(-1, 1, lin_size)
x2 = np.linspace(-1, 1, lin_size)
X1, X2 = np.meshgrid(x1, x2)
T = w_true[0] + w_true[1] * X1**2 + w_true[2] * X2**3 + np.random.normal(0, sigma, X1.shape)

# 2D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.contourf(X1, X2, T)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_title('Input Space: $x1 = [-1, -0.95, ..., 0.95, 1]$ x $x2 = [-1, -0.95, ..., 0.95, 1]$')
plt.show()

# 3D plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, T, cmap='viridis')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('t')
ax.set_title('Generated Data: $t = w_0 + w_1 x_1^2 + w_2 x_2^3 + \epsilon$')
plt.show()

# Step 2
x1_flat = X1.ravel()
x2_flat = X2.ravel()
t_flat = T.ravel()
X_all = np.vstack((x1_flat, x2_flat)).T

test_mask = (np.abs(x1_flat) > 0.31) & (np.abs(x2_flat) > 0.31)
X_test = X_all[test_mask]
t_test = t_flat[test_mask]
X_train = X_all[~test_mask]
t_train = t_flat[~test_mask]
t_test_noisy = t_test + np.random.normal(0, sigma, size=t_test.shape)

# Training and test data split plot
plt.figure(figsize=(8, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c='blue', label='Train', alpha=0.5)
plt.scatter(X_test[:, 0], X_test[:, 1], c='red', label='Test', alpha=0.5)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Train vs Test Split')
plt.legend()
plt.grid(True)
plt.show()

# Step 3
Phi_train = np.column_stack((np.ones(X_train.shape[0]), X_train[:, 0]**2, X_train[:, 1]**3))
Phi_test = np.column_stack((np.ones(X_test.shape[0]), X_test[:, 0]**2, X_test[:, 1]**3))

w_ml = np.linalg.inv(Phi_train.T @ Phi_train) @ Phi_train.T @ t_train
mean_pred_ML = Phi_test @ w_ml
mse_test = np.mean((mean_pred_ML - t_test_noisy) ** 2)

# ML predictions scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(t_test_noisy, mean_pred_ML, alpha=0.7)
plt.plot([t_test_noisy.min(), t_test_noisy.max()], [t_test_noisy.min(), t_test_noisy.max()], 'r--',
         label='Perfect Prediction')
plt.xlabel('True Target (Noisy)')
plt.ylabel('Predicted Target')
plt.title('ML Predictions on Test Data')
plt.legend()
plt.grid(True)
plt.show()

# Step 4
alpha = 0.3
beta = 1.0 / sigma**2
I = np.eye(Phi_train.shape[1])

SN_inv = alpha * I + beta * (Phi_train.T @ Phi_train)
SN = np.linalg.inv(SN_inv)
mN = beta * SN @ Phi_train.T @ t_train

mean_pred_bayes = Phi_test @ mN
var_pred_bayes = (1 / beta) + np.sum(Phi_test @ SN * Phi_test, axis=1)
std_pred_bayes = np.sqrt(var_pred_bayes)
mse_bayes = np.mean((mean_pred_bayes - t_test_noisy)**2)

# Bayesian predictions w/ error bars plot
plt.figure(figsize=(8, 6))
plt.errorbar(np.arange(len(t_test_noisy)), mean_pred_bayes, yerr=std_pred_bayes, fmt='o',
             label='Bayesian Prediction ± Std Dev', alpha=0.7)
plt.scatter(np.arange(len(t_test_noisy)), t_test_noisy, color='red', alpha=0.6, label='True (Noisy)')
plt.title('Bayesian Predictive Mean and Uncertainty')
plt.xlabel('Test Sample Index')
plt.ylabel('Target t')
plt.legend()
plt.grid(True)
plt.show()

# Step 5
print("--------- Model Comparison ---------\n")
print(f"ML Test MSE       : {mse_test:.4f}")
print(f"Bayesian Test MSE : {mse_bayes:.4f}")
print(f"Difference (ML - Bayesian): {mse_test - mse_bayes:.8f}\n")

# Plots from step 3 & 4 side by side for comparison
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(t_test_noisy, mean_pred_ML, alpha=0.7, label='ML')
plt.plot([t_test_noisy.min(), t_test_noisy.max()], [t_test_noisy.min(), t_test_noisy.max()], 'r--')
plt.title('ML Predictions')
plt.xlabel('True')
plt.ylabel('Predicted')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.errorbar(t_test_noisy, mean_pred_bayes, yerr=std_pred_bayes, fmt='o', alpha=0.7, label='Bayesian')
plt.plot([t_test_noisy.min(), t_test_noisy.max()], [t_test_noisy.min(), t_test_noisy.max()], 'r--')
plt.title('Bayesian Predictions')
plt.xlabel('True')
plt.ylabel('Predicted')
plt.grid(True)
plt.suptitle('Comparison: ML vs Bayesian')
plt.tight_layout(); plt.show()

# Step 6: Bayesian Predictions on Training Data
mean_pred_train = Phi_train @ mN
var_pred_train = (1 / beta) + np.sum(Phi_train @ SN * Phi_train, axis=1)
std_pred_train = np.sqrt(var_pred_train)
mse_train_bayes = np.mean((mean_pred_train - t_train)**2)

print("--------- Bayesian Performance Summary ---------\n")
print(f"Train MSE (Bayesian): {mse_train_bayes:.4f}")
print(f"Test MSE  (Bayesian): {mse_bayes:.4f}")
print(f"Avg. Variance on Train Predictions: {np.mean(var_pred_train):.4f}")
print(f"Avg. Variance on Test  Predictions: {np.mean(var_pred_bayes):.4f}")
print(f"Avg. Std Dev on Train Predictions: {np.mean(std_pred_train):.4f}")
print(f"Avg. Std Dev on Test  Predictions: {np.mean(std_pred_bayes):.4f}\n")