import numpy as np
import matplotlib.pyplot as plt

#-------------------------------- step 1 ---------------------------------
# Step 1: Define the input domain
x1 = np.linspace(-1, 1, 41)
x2 = np.linspace(-1, 1, 41)
X1, X2 = np.meshgrid(x1, x2)

# Step 2: Set model parameters and noise level
w = np.array([0, 2.5, -0.5])
sigma = 0.3

# Step 3: Compute target values with noise
T = w[0] + w[1] * X1**2 + w[2] * X2**3 + np.random.normal(0, sigma, X1.shape)

# Step 4: Plot the data
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, T, cmap='viridis')
plt.contour(X1, X2, T)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('t')
ax.set_title('Generated Data: $t = w_0 + w_1 x_1^2 + w_2 x_2^3 + \epsilon$')
# plt.contourf(X1, X2, T)
plt.show()

#-------------------------------- step 2 ---------------------------------
# Flatten the meshgrids for x1 and x2
x1_flat = X1.ravel()
x2_flat = X2.ravel()
t_flat = T.ravel()

# Combine into a single array for easy indexing
X_all = np.vstack((x1_flat, x2_flat)).T

# Boolean mask for test data: where both |x1| > 0.3 and |x2| > 0.3
test_mask = (np.abs(x1_flat) > 0.31) & (np.abs(x2_flat) > 0.31)

# Split data using the mask
X_test = X_all[test_mask]
t_test = t_flat[test_mask]

X_train = X_all[~test_mask]
t_train = t_flat[~test_mask]

# Add extra noise to the test outputs
extra_sigma = 0.2
t_test_noisy = t_test + np.random.normal(0, extra_sigma, size=t_test.shape)

# Optional: print dataset sizes
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Quick plot to visualize selection
plt.figure(figsize=(8, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c='blue', label='Training Data', alpha=0.5)
plt.scatter(X_test[:, 0], X_test[:, 1], c='red', label='Test Data', alpha=0.5)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Train vs Test Data Selection')
plt.legend()
plt.grid(True)
plt.show()

#-------------------------------- step 3 ---------------------------------
# Step 3: Define basis function mapping
def phi(x):
    # x is (N, 2), returns (N, 3)
    return np.column_stack((np.ones(x.shape[0]), x[:, 0]**2, x[:, 1]**3))

# Step 3a: Design matrices
Phi_train = phi(X_train)
Phi_test = phi(X_test)

# Step 3b: Compute ML weights (w_ML)
w_ml = np.linalg.inv(Phi_train.T @ Phi_train) @ Phi_train.T @ t_train

# Step 3c: Predict on test data
t_pred_test = Phi_test @ w_ml

# Step 3d: Evaluate MSE
mse_test = np.mean((t_pred_test - t_test_noisy)**2)

print(f"Maximum Likelihood Weights: {w_ml}")
print(f"Test MSE (ML): {mse_test:.4f}")

# Optional: Visualize predicted vs actual
plt.figure(figsize=(8, 6))
plt.scatter(t_test_noisy, t_pred_test, alpha=0.7)
plt.plot([t_test_noisy.min(), t_test_noisy.max()],
         [t_test_noisy.min(), t_test_noisy.max()], 'r--', label='Perfect Prediction')
plt.xlabel('True Target (Noisy)')
plt.ylabel('Predicted Target (ML)')
plt.title('Maximum Likelihood Predictions on Test Data')
plt.legend()
plt.grid(True)
plt.show()

#-------------------------------- step 4 ---------------------------------
# Step 4: Set hyperparameters
alpha = 0.7  # Prior precision (can try 0.3, 2.0, etc.)
beta = 1.0 / sigma**2  # Likelihood precision

# Step 4a: Compute posterior covariance (S_N) and mean (m_N)
I = np.eye(Phi_train.shape[1])
SN_inv = alpha * I + beta * (Phi_train.T @ Phi_train)
SN = np.linalg.inv(SN_inv)

mN = beta * SN @ Phi_train.T @ t_train

# Step 4b: Predictive mean and variance for test data
mean_pred_bayes = Phi_test @ mN
var_pred_bayes = (1 / beta) + np.sum(Phi_test @ SN * Phi_test, axis=1)
std_pred_bayes = np.sqrt(var_pred_bayes)

# Step 4c: MSE using predictive mean only (not full Bayesian spirit, but comparable to ML)
mse_bayes = np.mean((mean_pred_bayes - t_test_noisy)**2)

print(f"Bayesian Posterior Mean (mN):\n{mN}")
print(f"Bayesian Test MSE (Predictive Mean): {mse_bayes:.4f}")

# Visualize Bayesian predictions (with uncertainty)
plt.figure(figsize=(8, 6))
plt.errorbar(
    np.arange(len(t_test_noisy)),
    mean_pred_bayes,
    yerr=std_pred_bayes,
    fmt='o',
    label='Bayesian Prediction ± Std Dev',
    alpha=0.7
)
plt.scatter(np.arange(len(t_test_noisy)), t_test_noisy, color='red', label='True (Noisy)', alpha=0.6)
plt.title('Bayesian Predictive Mean and Uncertainty on Test Data')
plt.xlabel('Test Sample Index')
plt.ylabel('Target t')
plt.legend()
plt.grid(True)
plt.show()
