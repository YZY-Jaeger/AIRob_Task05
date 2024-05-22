import numpy as np

def initialize_arms(k=3):
    mus = np.random.uniform(-1, 1, k)
    sigmas = np.random.uniform(0.2, 1.2, k)
    return mus, sigmas**2  # Return variances

def simulate_round(mu, sigma, prediction):
    # Simulate feedback from the chosen arm
    feedback = np.random.normal(mu, np.sqrt(sigma))
    # Calculate prediction error
    error = abs(feedback - prediction)
    return feedback, error

def two_phase_bandit(mus, sigmas, T=300):
    k = len(mus)
    # Length of exploration phase
    T1 = T // 2
    # Initialize error and prediction tracking
    errors = np.zeros(T)
    predictions = np.zeros((T, k))
    feedbacks = np.zeros((T, k))
    
    print(f"Starting Exploration Phase for {T1} rounds")
    # Exploration phase: play each arm T1/k times
    for t in range(T1):
        arm = t % k
        # Use the mean of the arm as the prediction for simplicity
        prediction = mus[arm]
        feedback, error = simulate_round(mus[arm], sigmas[arm], prediction)
        # Store data
        errors[t] = error
        predictions[t, arm] = prediction
        feedbacks[t, arm] = feedback
        print(f"Round {t+1}: Exploring Arm {arm+1}, Predicted {prediction:.2f}, Observed {feedback:.2f}, Error {error:.2f}")
    
    # Update estimates based on exploration
    estimated_mus = np.mean(feedbacks[:T1], axis=0)
    print(f"Exploration Complete. Estimated Means: {estimated_mus}")
    
    # Exploitation phase: choose the best arm based on lowest error
    best_arm = np.argmin(np.abs(estimated_mus - mus))
    print(f"Starting Exploitation Phase with Arm {best_arm+1} based on lowest error estimation")
    
    for t in range(T1, T):
        prediction = estimated_mus[best_arm]
        feedback, error = simulate_round(mus[best_arm], sigmas[best_arm], prediction)
        errors[t] = error
        predictions[t, best_arm] = prediction
        print(f"Round {t+1}: Exploiting Arm {best_arm+1}, Predicted {prediction:.2f}, Observed {feedback:.2f}, Error {error:.2f}")
    
    return errors, predictions

# Run the experiment
mus, sigmas = initialize_arms()
print("Initialized Arms with Parameters:")
for i, (mu, sigma) in enumerate(zip(mus, np.sqrt(sigmas)), 1):
    print(f"Arm {i}: Mean = {mu:.2f}, Std Dev = {sigma:.2f}")

errors, predictions = two_phase_bandit(mus, sigmas)

print("Total Cumulated Prediction Error:", np.sum(errors))
