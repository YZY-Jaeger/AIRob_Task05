import numpy as np
import matplotlib.pyplot as plt

'''
randomly initiate the k = 3 normal distributions N (µi
, σ2i) with uniformly distributed
distribution parameters µi ∈ [−1, 1] and σi ∈ [0.2, 1.2] at the beginning of each new experiment run.
'''
def initialize_arms(k=3):
    mus = np.random.uniform(-1, 1, k)
    sigmas = np.random.uniform(0.2, 1.2, k)
    return mus, sigmas**2  # Return variances

def simulate_round(mu, sigma, prediction):
    feedback = np.random.normal(mu, np.sqrt(sigma))
    error = abs(feedback - prediction)
    return feedback, error

def two_phase_bandit(mus, sigmas, T=300, exploration_ratio=0.5):#by default 300 rounds and 50% exploration
    k = len(mus)
    T1 = int(T * exploration_ratio)
    errors = np.zeros(T)
    predictions = np.zeros((T, k))
    feedbacks = np.zeros((T, k))
    
    print(f"Starting Exploration Phase for {T1} rounds with exploration ratio of {exploration_ratio*100}%.")
    # Exploration phase
    for t in range(T1):
        arm = t % k
        prediction = mus[arm]
        feedback, error = simulate_round(mus[arm], sigmas[arm], prediction)
        errors[t] = error
        predictions[t, arm] = prediction
        feedbacks[t, arm] = feedback
        print(f"Round {t+1}: Arm {arm+1}, Prediction: {prediction:.2f}, Feedback: {feedback:.2f}, Error: {error:.2f}")

    estimated_mus = np.mean(feedbacks[:T1], axis=0)
    print("Estimated means after exploration:", estimated_mus)
    
    print(f"Starting Exploitation Phase with best arm selection based on estimated means.")
    # Exploitation phase
    best_arm = np.argmin(np.abs(estimated_mus - mus))
    for t in range(T1, T):
        prediction = estimated_mus[best_arm]
        feedback, error = simulate_round(mus[best_arm], sigmas[best_arm], prediction)
        errors[t] = error
        print(f"Round {t+1}: Best Arm {best_arm+1}, Prediction: {prediction:.2f}, Feedback: {feedback:.2f}, Error: {error:.2f}")

    return np.sum(errors)

def run_experiments(num_experiments=100, strategies=[0.25, 0.5, 0.75]):
    total_errors = {str(ratio): [] for ratio in strategies}
    for ratio in strategies:
        print(f"Testing strategy with {ratio*100}% exploration.")
        for _ in range(num_experiments):
            mus, sigmas = initialize_arms()
            total_error = two_phase_bandit(mus, sigmas, exploration_ratio=ratio)
            total_errors[str(ratio)].append(total_error)

    # Plot results
    plt.boxplot([total_errors[str(ratio)] for ratio in strategies])
    plt.xticks([1, 2, 3], ['25% Exploration', '50% Exploration', '75% Exploration'])
    plt.title('Comparison of Total Prediction Errors across Strategies')
    plt.ylabel('Total Prediction Error')


    # Save plot to disk
    plt.savefig('bandit_strategy_comparison.png')
    plt.show()
    return total_errors

# Run the experiments and plot the results
results = run_experiments()
