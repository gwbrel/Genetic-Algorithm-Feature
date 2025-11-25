# genetic_algorithm_feature_selection.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import random

class GeneticAlgorithmFeatureSelection:
    def __init__(self, population_size=50, generations=50, crossover_rate=0.8, 
                 mutation_rate=0.1, tournament_size=3, elitism_count=2):
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count
        self.best_individual = None
        self.best_fitness = 0
        self.fitness_history = []
        
    def initialize_population(self, chromosome_length):
        """Initialize random population of binary chromosomes"""
        population = np.random.randint(0, 2, (self.population_size, chromosome_length))
        return population
    
    def fitness_function(self, individual, X_train, X_val, y_train, y_val):
        """Calculate fitness (accuracy) for a given feature subset"""
        # Get selected features
        selected_features = np.where(individual == 1)[0]
        
        # If no features selected, return very low fitness
        if len(selected_features) == 0:
            return 0.0
        
        # Select features from datasets
        X_train_selected = X_train[:, selected_features]
        X_val_selected = X_val[:, selected_features]
        
        # Train KNN classifier
        knn = KNeighborsClassifier()
        knn.fit(X_train_selected, y_train)
        
        # Predict and calculate accuracy
        y_pred = knn.predict(X_val_selected)
        accuracy = accuracy_score(y_val, y_pred)
        
        # Optional: Add penalty for too many features to encourage smaller subsets
        feature_penalty = 0.01 * (len(selected_features) / X_train.shape[1])
        fitness = accuracy - feature_penalty
        
        return fitness
    
    def tournament_selection(self, population, fitness_values):
        """Select parents using tournament selection"""
        selected_parents = []
        
        for _ in range(2):  # Select 2 parents
            # Randomly select tournament participants
            tournament_indices = np.random.choice(
                len(population), self.tournament_size, replace=False
            )
            tournament_fitness = [fitness_values[i] for i in tournament_indices]
            
            # Select the best from tournament
            winner_index = tournament_indices[np.argmax(tournament_fitness)]
            selected_parents.append(population[winner_index])
        
        return selected_parents[0], selected_parents[1]
    
    def crossover(self, parent1, parent2):
        """Perform single-point crossover"""
        if random.random() < self.crossover_rate:
            # Choose crossover point
            crossover_point = random.randint(1, len(parent1) - 1)
            
            # Create offspring
            child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
            
            return child1, child2
        else:
            return parent1.copy(), parent2.copy()
    
    def mutation(self, individual):
        """Perform bit-flip mutation"""
        mutated_individual = individual.copy()
        
        for i in range(len(mutated_individual)):
            if random.random() < self.mutation_rate:
                mutated_individual[i] = 1 - mutated_individual[i]  # Flip bit
        
        return mutated_individual
    
    def run(self, X_train, X_val, y_train, y_val):
        """Run the genetic algorithm"""
        chromosome_length = X_train.shape[1]
        population = self.initialize_population(chromosome_length)
        
        print("Starting Genetic Algorithm...")
        start_time = time.time()
        
        for generation in range(self.generations):
            # Evaluate fitness for each individual
            fitness_values = []
            for individual in population:
                fitness = self.fitness_function(individual, X_train, X_val, y_train, y_val)
                fitness_values.append(fitness)
            
            # Update best individual
            current_best_fitness = max(fitness_values)
            current_best_index = np.argmax(fitness_values)
            
            if current_best_fitness > self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_individual = population[current_best_index].copy()
            
            self.fitness_history.append(current_best_fitness)
            
            # Create new population
            new_population = []
            
            # Apply elitism
            elite_indices = np.argsort(fitness_values)[-self.elitism_count:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # Fill the rest of the population
            while len(new_population) < self.population_size:
                # Selection
                parent1, parent2 = self.tournament_selection(population, fitness_values)
                
                # Crossover
                child1, child2 = self.crossover(parent1, parent2)
                
                # Mutation
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                
                # Add children to new population
                if len(new_population) < self.population_size:
                    new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            population = np.array(new_population)
            
            if (generation + 1) % 10 == 0:
                selected_features = np.sum(self.best_individual)
                print(f"Generation {generation + 1}: Best Fitness = {self.best_fitness:.4f}, "
                      f"Selected Features = {selected_features}")
        
        execution_time = time.time() - start_time
        print(f"\nGenetic Algorithm completed in {execution_time:.2f} seconds")
        
        return self.best_individual, execution_time

def load_and_preprocess_data():
    """Load and preprocess the Breast Cancer dataset"""
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split data: 60% training, 20% validation, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp  # 0.25 * 0.8 = 0.2
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def evaluate_with_all_features(X_train, X_val, X_test, y_train, y_val, y_test):
    """Evaluate KNN classifier using all features"""
    print("\nEvaluating with all features...")
    start_time = time.time()
    
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    
    # Use validation set for consistency (though in practice we'd use test set only once)
    y_pred_val = knn.predict(X_val)
    accuracy_val = accuracy_score(y_val, y_pred_val)
    
    # Final evaluation on test set
    y_pred_test = knn.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    
    execution_time = time.time() - start_time
    
    print(f"All Features - Validation Accuracy: {accuracy_val:.4f}")
    print(f"All Features - Test Accuracy: {accuracy_test:.4f}")
    print(f"Execution Time: {execution_time:.2f} seconds")
    
    return accuracy_test, execution_time

def evaluate_ga_features(best_individual, X_train, X_val, X_test, y_train, y_val, y_test):
    """Evaluate the best feature subset found by GA"""
    print("\nEvaluating GA-selected features...")
    
    # Get selected features
    selected_features = np.where(best_individual == 1)[0]
    num_selected = len(selected_features)
    
    print(f"Number of selected features: {num_selected}")
    print(f"Selected feature indices: {selected_features}")
    
    # Select features from datasets
    X_train_selected = X_train[:, selected_features]
    X_val_selected = X_val[:, selected_features]
    X_test_selected = X_test[:, selected_features]
    
    # Train and evaluate
    knn = KNeighborsClassifier()
    knn.fit(X_train_selected, y_train)
    
    # Validation accuracy (used in fitness function)
    y_pred_val = knn.predict(X_val_selected)
    accuracy_val = accuracy_score(y_val, y_pred_val)
    
    # Test accuracy
    y_pred_test = knn.predict(X_test_selected)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    
    print(f"GA Features - Validation Accuracy: {accuracy_val:.4f}")
    print(f"GA Features - Test Accuracy: {accuracy_test:.4f}")
    
    return accuracy_test, num_selected

def plot_results(fitness_history, ga_accuracy, all_features_accuracy):
    """Plot the results of the genetic algorithm"""
    plt.figure(figsize=(12, 4))
    
    # Plot fitness evolution
    plt.subplot(1, 2, 1)
    plt.plot(fitness_history)
    plt.title('Genetic Algorithm Fitness Evolution')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness (Validation Accuracy)')
    plt.grid(True)
    
    # Plot accuracy comparison
    plt.subplot(1, 2, 2)
    methods = ['All Features', 'GA Selected']
    accuracies = [all_features_accuracy, ga_accuracy]
    bars = plt.bar(methods, accuracies, color=['skyblue', 'lightcoral'])
    plt.title('Test Set Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, accuracy in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{accuracy:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('ga_feature_selection_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run the complete experiment"""
    print("=" * 60)
    print("Feature Selection using Genetic Algorithms")
    print("Breast Cancer Dataset - KNN Classifier")
    print("=" * 60)
    
    # Load and preprocess data
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data()
    
    print(f"Dataset shapes:")
    print(f"Training: {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print(f"Test: {X_test.shape}")
    
    # Evaluate with all features
    all_features_accuracy, all_features_time = evaluate_with_all_features(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    # Run Genetic Algorithm for feature selection
    ga = GeneticAlgorithmFeatureSelection(
        population_size=30,
        generations=40,
        crossover_rate=0.8,
        mutation_rate=0.05,
        tournament_size=3,
        elitism_count=2
    )
    
    best_individual, ga_time = ga.run(X_train, X_val, y_train, y_val)
    
    # Evaluate GA-selected features
    ga_accuracy, num_selected_features = evaluate_ga_features(
        best_individual, X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    # Display comparative results
    print("\n" + "=" * 60)
    print("COMPARATIVE RESULTS")
    print("=" * 60)
    print(f"{'Method':<20} {'# Features':<12} {'Test Accuracy':<15} {'Execution Time (s)':<18}")
    print("-" * 60)
    print(f"{'Without Selection':<20} {30:<12} {all_features_accuracy:<15.4f} {all_features_time:<18.2f}")
    print(f"{'GA':<20} {num_selected_features:<12} {ga_accuracy:<15.4f} {ga_time:<18.2f}")
    
    # Plot results
    plot_results(ga.fitness_history, ga_accuracy, all_features_accuracy)
    
    return {
        'all_features_accuracy': all_features_accuracy,
        'all_features_time': all_features_time,
        'ga_accuracy': ga_accuracy,
        'ga_time': ga_time,
        'num_selected_features': num_selected_features,
        'best_individual': best_individual
    }

if __name__ == "__main__":
    results = main()