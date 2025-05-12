import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# ----- Step 1: Data Generation and 3D Visualization -----
# =============================================================================
# Generate a 2D grid for the input space, then compute the target values
# using the model t = w0 + w1*x1^2 + w2*x2^3 + noise.
# The weights correspond to the "true" model parameters used for data generation.

# Define the input ranges for x1 and x2 (matching the specification from Eq. 38)
x1_values = np.linspace(-1, 1, 41)
x2_values = np.linspace(-1, 1, 41)
X1_grid, X2_grid = np.meshgrid(x1_values, x2_values)

# Define the true model parameters and noise level (sigma)
true_weights = np.array([0, 2.5, -0.5])  # [w0, w1, w2]
noise_sigma = 0.3

# Compute the target outputs using the model: t = w0 + w1*x1^2 + w2*x2^3 + noise
T_generated = true_weights[0] + true_weights[1] * X1_grid**2 + true_weights[2] * X2_grid**3
T_generated += np.random.normal(0, noise_sigma, X1_grid.shape)

# 3D surface plot with contour overlay of the generated data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1_grid, X2_grid, T_generated, cmap='viridis', alpha=0.9)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('t')
ax.set_title('Generated Data: t = w0 + w1*x1² + w2*x2³ + ε')
plt.show()

# 2D contourf plot
fig = plt.figure()
ax_2 = fig.add_subplot(111)
ax_2.contourf(X1_grid, X2_grid, T_generated, cmap='viridis')
ax_2.set_xlabel('x1')
ax_2.set_ylabel('x2')
ax_2.set_title('Input space: x1 = [-1, 0.95, ..., 0.95, 1] x x2 = [-1, 0.95, ..., 0.95, 1]')
plt.show()


# =============================================================================
# ----- Step 2: Data Reshaping, Train-Test Split, and Adding Extra Noise -----
# =============================================================================
# Flatten the 2D grid into a list of input points and corresponding noisy targets.
# Then, split the data into training and test sets based on the lab task:
# test points are those where |x1| > 0.31 and |x2| > 0.31.
# Additionally, add extra noise to the test outputs (t_extra_noise) for realism.

# Flatten grid arrays
x1_flat = X1_grid.ravel()
x2_flat = X2_grid.ravel()
targets_flat = T_generated.ravel()

# Combine x1 and x2 into a single input feature array
inputs_all = np.vstack((x1_flat, x2_flat)).T

# Define a boolean mask for test data based on the specified conditions.
test_mask = (np.abs(x1_flat) > 0.31) & (np.abs(x2_flat) > 0.31)

# Split the data into training and test subsets
inputs_test = inputs_all[test_mask]
targets_test = targets_flat[test_mask]
inputs_train = inputs_all[~test_mask]
targets_train = targets_flat[~test_mask]

# Add extra noise to the test targets to simulate the specified "extra noise"
extra_noise_sigma = 0.2
targets_test_noisy = targets_test + np.random.normal(0, extra_noise_sigma, size=targets_test.shape)

print(f"Training samples: {len(inputs_train)}")
print(f"Testing samples: {len(inputs_test)}")

# Quick scatter plot to show which points went into training vs. test
plt.figure(figsize=(8, 6))
plt.scatter(inputs_train[:, 0], inputs_train[:, 1], c='blue', label='Training Data', alpha=0.6)
plt.scatter(inputs_test[:, 0], inputs_test[:, 1], c='red', label='Test Data', alpha=0.6)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Train vs. Test Data Selection')
plt.legend()
plt.grid(True)
plt.show()


# =============================================================================
# ----- Step 3: Constructing the Design Matrix and Computing ML Weights -----
# =============================================================================
# Define a basis function mapping according to the model in Eq. 38. Here,
# the mapping is: φ(x) = [1, x1², x2³]. Then create the design matrices for
# the training and test inputs, compute the maximum likelihood (ML) weights,
# and evaluate the ML predictions against the test set.

def basis_function_phi(X):
    """
    Mapping to the design space:
      For each input vector [x1, x2] return [1, x1², x2³]
    """
    return np.column_stack((np.ones(X.shape[0]), X[:, 0]**2, X[:, 1]**3))

# Construct design matrices for training and test data
Phi_train = basis_function_phi(inputs_train)
Phi_test = basis_function_phi(inputs_test)

# Compute the ML estimator for weights: w_ML = (Φ^TΦ)⁻¹Φ^T t_train
weights_ml = np.linalg.inv(Phi_train.T @ Phi_train) @ (Phi_train.T @ targets_train)

# Predict the target values for test data using ML estimates: t_pred_ML = Φ_test * weights_ml
targets_pred_ml = Phi_test @ weights_ml

# Evaluate the ML model using Mean Squared Error (MSE)
mse_ml = np.mean((targets_pred_ml - targets_test_noisy)**2)

print(f"ML Weights: {weights_ml}")
print(f"Test MSE (ML): {mse_ml:.4f}")

# Scatter plot to show ML predictions compared with noisy test targets
plt.figure(figsize=(8, 6))
plt.scatter(targets_test_noisy, targets_pred_ml, alpha=0.7)
plt.plot([targets_test_noisy.min(), targets_test_noisy.max()],
         [targets_test_noisy.min(), targets_test_noisy.max()], 'r--', label='Perfect Prediction')
plt.xlabel('True Noisy Test Targets')
plt.ylabel('ML Predicted Targets')
plt.title('ML Predictions on Test Data')
plt.legend()
plt.grid(True)
plt.show()


# =============================================================================
# ----- Step 4: Bayesian Posterior Computation and Predictive Uncertainty -----
# =============================================================================
# Set the hyperparameters for the Bayesian model (α for the prior precision, β for the
# likelihood precision). Then compute the posterior over the weight parameters
# (posterior mean m_N and covariance S_N), and use these to calculate the Bayesian
# predictive distribution (its mean and variance) on the test set.

# Set hyperparameters: α (prior precision) and β (likelihood precision = 1/σ²)
alpha_value = 0.7
beta_value = 1.0 / (noise_sigma**2)

# Compute the posterior covariance matrix: S_N = (αI + β Φ_train^T Φ_train)⁻¹
identity_matrix = np.eye(Phi_train.shape[1])
posterior_covariance = np.linalg.inv(alpha_value * identity_matrix + beta_value * (Phi_train.T @ Phi_train))

# Compute the posterior mean vector: m_N = β S_N Φ_train^T t_train
posterior_mean = beta_value * posterior_covariance @ (Phi_train.T @ targets_train)

# Bayesian predictive mean for each test point: μ_pred = Φ_test * m_N
bayes_pred_mean = Phi_test @ posterior_mean

# Bayesian predictive variance: σ_pred² = 1/β + diag(Φ_test * S_N * Φ_test^T)
bayes_pred_variance = (1.0 / beta_value) + np.sum(Phi_test @ posterior_covariance * Phi_test, axis=1)
bayes_pred_std = np.sqrt(bayes_pred_variance)

# Calculate Bayesian test MSE (using the predictive mean)
mse_bayes = np.mean((bayes_pred_mean - targets_test_noisy)**2)

print(f"Bayesian Posterior Mean: {posterior_mean}")
print(f"Test MSE (Bayesian Predictive Mean): {mse_bayes:.4f}")

# Plot the Bayesian predictive mean with error bars (± standard deviation)
plt.figure(figsize=(8, 6))
plt.errorbar(np.arange(len(targets_test_noisy)), bayes_pred_mean, yerr=bayes_pred_std,
             fmt='o', capsize=5, label='Bayesian Prediction ± Std Dev', color='blue', alpha=0.7)
plt.scatter(np.arange(len(targets_test_noisy)), targets_test_noisy, color='red', label='True Noisy Targets', alpha=0.6)
plt.xlabel('Test Sample Index')
plt.ylabel('Target Value')
plt.title('Bayesian Predictive Mean and Uncertainty on Test Data')
plt.legend()
plt.grid(True)
plt.show()


# =============================================================================
# ----- Step 5: Comparison Between ML and Bayesian Approaches -----
# =============================================================================
# Compare the performance of the ML and Bayesian models by computing and printing
# their Mean Squared Errors (MSE) and by visualizing the residuals (error histograms)
# and a scatter plot of predicted versus true values for each approach.

# Compute residuals for the ML and Bayesian predictions
residuals_ml = targets_test_noisy - targets_pred_ml
residuals_bayes = targets_test_noisy - bayes_pred_mean

print("ML Test MSE:", mse_ml)
print("Bayesian Test MSE (Predictive Mean):", mse_bayes)

# Scatter plot comparing the predictions directly to the true noisy targets
plt.figure(figsize=(8, 6))
plt.plot(targets_test_noisy, targets_pred_ml, 'bo', label='ML Predictions')
plt.plot(targets_test_noisy, bayes_pred_mean, 'ro', label='Bayesian Predictions')
plt.plot([targets_test_noisy.min(), targets_test_noisy.max()],
         [targets_test_noisy.min(), targets_test_noisy.max()], 'k--', label='Ideal (y=x)')
plt.xlabel('True Noisy Test Targets')
plt.ylabel('Predicted Target')
plt.title('Comparison: ML vs Bayesian Predictions')
plt.legend()
plt.grid(True)
plt.show()


# =============================================================================
# ----- Step 6: Overlaying ML Deterministic Predictions on Bayesian Predictive Distributions -----
# =============================================================================
# Overlay the deterministic ML predictions on the Bayesian predictive plot.
# This visualization combines the Bayesian predictive mean (with error bars representing
# uncertainty) and the ML predictions, allowing for a direct comparison on test data.

# Create an index for plotting test samples
test_sample_indices = np.arange(len(targets_test_noisy))

plt.figure(figsize=(10, 6))

# Plot Bayesian predictive mean with error bars (uncertainty)
plt.errorbar(test_sample_indices, bayes_pred_mean, yerr=bayes_pred_std,
             fmt='o', capsize=5, label='Bayesian Prediction ± Std Dev', color='blue', alpha=0.7)

# Overlay ML deterministic predictions
plt.plot(test_sample_indices, targets_pred_ml, 's', markersize=8, linestyle='None',
         label='ML Deterministic Prediction', color='green')

# Plot the true noisy test targets for reference
plt.scatter(test_sample_indices, targets_test_noisy, color='red', label='True Noisy Targets', alpha=0.6)

plt.xlabel('Test Sample Index')
plt.ylabel('Target Value')
plt.title('Overlay: ML Predictions on Bayesian Predictive Distributions')
plt.legend()
plt.grid(True)
plt.show()
