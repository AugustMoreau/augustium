//! Machine Learning module for Augustium smart contracts
//! 
//! This module provides ML capabilities including neural networks,
//! linear regression, decision trees, and data preprocessing for blockchain applications.

// Core types not needed for ML module
use crate::error::{Result, CompilerError};
use std::collections::HashMap;
// fmt not needed
use serde::{Serialize, Deserialize};

/// Neural Network implementation for smart contracts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralNetwork {
    layers: Vec<Layer>,
    learning_rate: f64,
    activation_function: ActivationFunction,
}

/// Layer in a neural network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer {
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
    neurons: usize,
}

/// Supported activation functions
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ActivationFunction {
    Sigmoid,
    ReLU,
    Tanh,
    Linear,
}

/// Machine Learning Model trait
pub trait MLModel {
    type Input;
    type Output;
    
    fn train(&mut self, inputs: &[Self::Input], targets: &[Self::Output]) -> Result<()>;
    fn predict(&self, input: &Self::Input) -> Result<Self::Output>;
    fn save_model(&self) -> Result<Vec<u8>>;
    fn load_model(&mut self, data: &[u8]) -> Result<()>;
}

/// Linear Regression model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearRegression {
    weights: Vec<f64>,
    bias: f64,
    learning_rate: f64,
}

/// Decision Tree node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTreeNode {
    feature_index: Option<usize>,
    threshold: Option<f64>,
    left: Option<Box<DecisionTreeNode>>,
    right: Option<Box<DecisionTreeNode>>,
    prediction: Option<f64>,
}

/// Decision Tree model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTree {
    root: Option<Box<DecisionTreeNode>>,
    max_depth: usize,
    min_samples_split: usize,
}

/// Data preprocessing utilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPreprocessor {
    feature_means: Vec<f64>,
    feature_stds: Vec<f64>,
    is_fitted: bool,
}

/// ML Dataset for training
#[derive(Debug, Clone)]
pub struct MLDataset {
    features: Vec<Vec<f64>>,
    targets: Vec<f64>,
    feature_names: Vec<String>,
}

/// Model evaluation metrics
#[derive(Debug, Clone)]
pub struct ModelMetrics {
    accuracy: f64,
    precision: f64,
    recall: f64,
    f1_score: f64,
    mse: f64,
    mae: f64,
}

impl NeuralNetwork {
    /// Create a new neural network
    pub fn new(layer_sizes: &[usize], learning_rate: f64, activation: ActivationFunction) -> Self {
        let mut layers = Vec::new();
        
        for i in 1..layer_sizes.len() {
            let input_size = layer_sizes[i - 1];
            let output_size = layer_sizes[i];
            
            let mut weights = Vec::new();
            for _ in 0..output_size {
                let mut row = Vec::new();
                for _ in 0..input_size {
                    // Xavier initialization
                    let range = (6.0 / (input_size + output_size) as f64).sqrt();
                    row.push((fastrand::f64() - 0.5) * 2.0 * range);
                }
                weights.push(row);
            }
            
            let biases = vec![0.0; output_size];
            
            layers.push(Layer {
                weights,
                biases,
                neurons: output_size,
            });
        }
        
        Self {
            layers,
            learning_rate,
            activation_function: activation,
        }
    }
    
    /// Forward propagation
    pub fn forward(&self, input: &[f64]) -> Result<Vec<f64>> {
        let mut current_input = input.to_vec();
        
        for layer in &self.layers {
            let mut output = Vec::new();
            
            for (neuron_idx, neuron_weights) in layer.weights.iter().enumerate() {
                let mut sum = layer.biases[neuron_idx];
                
                for (weight_idx, &weight) in neuron_weights.iter().enumerate() {
                    if weight_idx < current_input.len() {
                        sum += weight * current_input[weight_idx];
                    }
                }
                
                output.push(self.activate(sum));
            }
            
            current_input = output;
        }
        
        Ok(current_input)
    }
    
    /// Apply activation function
    fn activate(&self, x: f64) -> f64 {
        match self.activation_function {
            ActivationFunction::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            ActivationFunction::ReLU => x.max(0.0),
            ActivationFunction::Tanh => x.tanh(),
            ActivationFunction::Linear => x,
        }
    }
    
    /// Derivative of activation function
    fn activate_derivative(&self, x: f64) -> f64 {
        match self.activation_function {
            ActivationFunction::Sigmoid => {
                let s = self.activate(x);
                s * (1.0 - s)
            },
            ActivationFunction::ReLU => if x > 0.0 { 1.0 } else { 0.0 },
            ActivationFunction::Tanh => 1.0 - x.tanh().powi(2),
            ActivationFunction::Linear => 1.0,
        }
    }
}

impl MLModel for NeuralNetwork {
    type Input = Vec<f64>;
    type Output = Vec<f64>;
    
    fn train(&mut self, inputs: &[Self::Input], targets: &[Self::Output]) -> Result<()> {
        for (input, target) in inputs.iter().zip(targets.iter()) {
            // Forward pass
            let prediction = self.forward(input)?;
            
            // Backward pass (simplified)
            let output_error: Vec<f64> = prediction.iter()
                .zip(target.iter())
                .map(|(pred, targ)| pred - targ)
                .collect();
            
            // Update weights (simplified gradient descent)
            if let Some(last_layer) = self.layers.last_mut() {
                for (neuron_idx, error) in output_error.iter().enumerate() {
                    if neuron_idx < last_layer.weights.len() {
                        for (weight_idx, weight) in last_layer.weights[neuron_idx].iter_mut().enumerate() {
                            if weight_idx < input.len() {
                                *weight -= self.learning_rate * error * input[weight_idx];
                            }
                        }
                        last_layer.biases[neuron_idx] -= self.learning_rate * error;
                    }
                }
            }
        }
        
        Ok(())
    }
    
    fn predict(&self, input: &Self::Input) -> Result<Self::Output> {
        self.forward(input)
    }
    
    fn save_model(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(&self)
            .map_err(|e| CompilerError::InternalError(format!("Failed to serialize model: {}", e)))
    }
    
    fn load_model(&mut self, data: &[u8]) -> Result<()> {
        let model: NeuralNetwork = serde_json::from_slice(data)
            .map_err(|e| CompilerError::InternalError(format!("Failed to deserialize model: {}", e)))?;
        *self = model;
        Ok(())
    }
}

impl LinearRegression {
    /// Create a new linear regression model
    pub fn new(features: usize, learning_rate: f64) -> Self {
        Self {
            weights: vec![0.0; features],
            bias: 0.0,
            learning_rate,
        }
    }
    
    /// Predict using linear regression
    pub fn predict_value(&self, features: &[f64]) -> f64 {
        let mut prediction = self.bias;
        for (weight, feature) in self.weights.iter().zip(features.iter()) {
            prediction += weight * feature;
        }
        prediction
    }
}

impl MLModel for LinearRegression {
    type Input = Vec<f64>;
    type Output = f64;
    
    fn train(&mut self, inputs: &[Self::Input], targets: &[Self::Output]) -> Result<()> {
        let n = inputs.len() as f64;
        
        for (input, &target) in inputs.iter().zip(targets.iter()) {
            let prediction = self.predict_value(input);
            let error = prediction - target;
            
            // Update weights
            for (weight, &feature) in self.weights.iter_mut().zip(input.iter()) {
                *weight -= self.learning_rate * error * feature / n;
            }
            
            // Update bias
            self.bias -= self.learning_rate * error / n;
        }
        
        Ok(())
    }
    
    fn predict(&self, input: &Self::Input) -> Result<Self::Output> {
        Ok(self.predict_value(input))
    }
    
    fn save_model(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(&self)
            .map_err(|e| CompilerError::InternalError(format!("Failed to serialize model: {}", e)))
    }
    
    fn load_model(&mut self, data: &[u8]) -> Result<()> {
        let model: LinearRegression = serde_json::from_slice(data)
            .map_err(|e| CompilerError::InternalError(format!("Failed to deserialize model: {}", e)))?;
        *self = model;
        Ok(())
    }
}

impl DecisionTree {
    /// Create a new decision tree
    pub fn new(max_depth: usize, min_samples_split: usize) -> Self {
        Self {
            root: None,
            max_depth,
            min_samples_split,
        }
    }
    
    /// Build the decision tree
    pub fn fit(&mut self, features: &[Vec<f64>], targets: &[f64]) -> Result<()> {
        self.root = Some(Box::new(self.build_tree(features, targets, 0)?));
        Ok(())
    }
    
    /// Recursively build the tree
    fn build_tree(&self, features: &[Vec<f64>], targets: &[f64], depth: usize) -> Result<DecisionTreeNode> {
        // Base cases
        if depth >= self.max_depth || features.len() < self.min_samples_split {
            let prediction = targets.iter().sum::<f64>() / targets.len() as f64;
            return Ok(DecisionTreeNode {
                feature_index: None,
                threshold: None,
                left: None,
                right: None,
                prediction: Some(prediction),
            });
        }
        
        // Find best split (simplified)
        let mut best_feature = 0;
        let mut best_threshold = 0.0;
        let mut best_score = f64::INFINITY;
        
        for feature_idx in 0..features[0].len() {
            let mut values: Vec<f64> = features.iter().map(|f| f[feature_idx]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            for &threshold in &values {
                let score = self.calculate_split_score(features, targets, feature_idx, threshold);
                if score < best_score {
                    best_score = score;
                    best_feature = feature_idx;
                    best_threshold = threshold;
                }
            }
        }
        
        // Split data
        let (left_features, left_targets, right_features, right_targets) = 
            self.split_data(features, targets, best_feature, best_threshold);
        
        // Create child nodes
        let left = if !left_features.is_empty() {
            Some(Box::new(self.build_tree(&left_features, &left_targets, depth + 1)?))
        } else {
            None
        };
        
        let right = if !right_features.is_empty() {
            Some(Box::new(self.build_tree(&right_features, &right_targets, depth + 1)?))
        } else {
            None
        };
        
        Ok(DecisionTreeNode {
            feature_index: Some(best_feature),
            threshold: Some(best_threshold),
            left,
            right,
            prediction: None,
        })
    }
    
    /// Calculate split score (MSE for regression)
    fn calculate_split_score(&self, features: &[Vec<f64>], targets: &[f64], feature_idx: usize, threshold: f64) -> f64 {
        let (_, left_targets, _, right_targets) = self.split_data(features, targets, feature_idx, threshold);
        
        let left_mse = if !left_targets.is_empty() {
            let mean = left_targets.iter().sum::<f64>() / left_targets.len() as f64;
            left_targets.iter().map(|&t| (t - mean).powi(2)).sum::<f64>() / left_targets.len() as f64
        } else {
            0.0
        };
        
        let right_mse = if !right_targets.is_empty() {
            let mean = right_targets.iter().sum::<f64>() / right_targets.len() as f64;
            right_targets.iter().map(|&t| (t - mean).powi(2)).sum::<f64>() / right_targets.len() as f64
        } else {
            0.0
        };
        
        let total_samples = targets.len() as f64;
        let left_weight = left_targets.len() as f64 / total_samples;
        let right_weight = right_targets.len() as f64 / total_samples;
        
        left_weight * left_mse + right_weight * right_mse
    }
    
    /// Split data based on feature and threshold
    fn split_data(&self, features: &[Vec<f64>], targets: &[f64], feature_idx: usize, threshold: f64) 
        -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>, Vec<f64>) {
        let mut left_features = Vec::new();
        let mut left_targets = Vec::new();
        let mut right_features = Vec::new();
        let mut right_targets = Vec::new();
        
        for (i, feature_row) in features.iter().enumerate() {
            if feature_row[feature_idx] <= threshold {
                left_features.push(feature_row.clone());
                left_targets.push(targets[i]);
            } else {
                right_features.push(feature_row.clone());
                right_targets.push(targets[i]);
            }
        }
        
        (left_features, left_targets, right_features, right_targets)
    }
    
    /// Predict using the decision tree
    pub fn predict_value(&self, features: &[f64]) -> Result<f64> {
        if let Some(ref root) = self.root {
            self.predict_node(root, features)
        } else {
            Err(CompilerError::InternalError("Decision tree not trained".to_string()))
        }
    }
    
    /// Predict using a specific node
    fn predict_node(&self, node: &DecisionTreeNode, features: &[f64]) -> Result<f64> {
        if let Some(prediction) = node.prediction {
            return Ok(prediction);
        }
        
        if let (Some(feature_idx), Some(threshold)) = (node.feature_index, node.threshold) {
            if features[feature_idx] <= threshold {
                if let Some(ref left) = node.left {
                    return self.predict_node(left, features);
                }
            } else {
                if let Some(ref right) = node.right {
                    return self.predict_node(right, features);
                }
            }
        }
        
        Err(CompilerError::InternalError("Invalid decision tree structure".to_string()))
    }
}

impl MLModel for DecisionTree {
    type Input = Vec<f64>;
    type Output = f64;
    
    fn train(&mut self, inputs: &[Self::Input], targets: &[Self::Output]) -> Result<()> {
        self.fit(inputs, targets)
    }
    
    fn predict(&self, input: &Self::Input) -> Result<Self::Output> {
        self.predict_value(input)
    }
    
    fn save_model(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(&self)
            .map_err(|e| CompilerError::InternalError(format!("Failed to serialize model: {}", e)))
    }
    
    fn load_model(&mut self, data: &[u8]) -> Result<()> {
        let model: DecisionTree = serde_json::from_slice(data)
            .map_err(|e| CompilerError::InternalError(format!("Failed to deserialize model: {}", e)))?;
        *self = model;
        Ok(())
    }
}

impl DataPreprocessor {
    /// Create a new data preprocessor
    pub fn new() -> Self {
        Self {
            feature_means: Vec::new(),
            feature_stds: Vec::new(),
            is_fitted: false,
        }
    }
    
    /// Fit the preprocessor to data
    pub fn fit(&mut self, data: &[Vec<f64>]) -> Result<()> {
        if data.is_empty() {
            return Err(CompilerError::InternalError("Cannot fit on empty data".to_string()));
        }
        
        let n_features = data[0].len();
        let n_samples = data.len() as f64;
        
        // Calculate means
        self.feature_means = vec![0.0; n_features];
        for sample in data {
            for (i, &value) in sample.iter().enumerate() {
                self.feature_means[i] += value;
            }
        }
        for mean in &mut self.feature_means {
            *mean /= n_samples;
        }
        
        // Calculate standard deviations
        self.feature_stds = vec![0.0; n_features];
        for sample in data {
            for (i, &value) in sample.iter().enumerate() {
                self.feature_stds[i] += (value - self.feature_means[i]).powi(2);
            }
        }
        for std in &mut self.feature_stds {
            *std = (*std / n_samples).sqrt();
            if *std == 0.0 {
                *std = 1.0; // Avoid division by zero
            }
        }
        
        self.is_fitted = true;
        Ok(())
    }
    
    /// Transform data using fitted parameters
    pub fn transform(&self, data: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        if !self.is_fitted {
            return Err(CompilerError::InternalError("Preprocessor not fitted".to_string()));
        }
        
        let mut transformed = Vec::new();
        for sample in data {
            let mut transformed_sample = Vec::new();
            for (i, &value) in sample.iter().enumerate() {
                let normalized = (value - self.feature_means[i]) / self.feature_stds[i];
                transformed_sample.push(normalized);
            }
            transformed.push(transformed_sample);
        }
        
        Ok(transformed)
    }
    
    /// Fit and transform in one step
    pub fn fit_transform(&mut self, data: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        self.fit(data)?;
        self.transform(data)
    }
}

impl MLDataset {
    /// Create a new ML dataset
    pub fn new(features: Vec<Vec<f64>>, targets: Vec<f64>, feature_names: Vec<String>) -> Self {
        Self {
            features,
            targets,
            feature_names,
        }
    }
    
    /// Split dataset into training and testing sets
    pub fn train_test_split(&self, test_ratio: f64) -> (MLDataset, MLDataset) {
        let test_size = (self.features.len() as f64 * test_ratio) as usize;
        let train_size = self.features.len() - test_size;
        
        let train_features = self.features[..train_size].to_vec();
        let train_targets = self.targets[..train_size].to_vec();
        let test_features = self.features[train_size..].to_vec();
        let test_targets = self.targets[train_size..].to_vec();
        
        let train_dataset = MLDataset::new(train_features, train_targets, self.feature_names.clone());
        let test_dataset = MLDataset::new(test_features, test_targets, self.feature_names.clone());
        
        (train_dataset, test_dataset)
    }
    
    /// Get dataset statistics
    pub fn describe(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();
        
        stats.insert("samples".to_string(), self.features.len() as f64);
        stats.insert("features".to_string(), self.features[0].len() as f64);
        
        // Target statistics
        let target_mean = self.targets.iter().sum::<f64>() / self.targets.len() as f64;
        let target_std = (self.targets.iter()
            .map(|&t| (t - target_mean).powi(2))
            .sum::<f64>() / self.targets.len() as f64).sqrt();
        
        stats.insert("target_mean".to_string(), target_mean);
        stats.insert("target_std".to_string(), target_std);
        stats.insert("target_min".to_string(), self.targets.iter().fold(f64::INFINITY, |a, &b| a.min(b)));
        stats.insert("target_max".to_string(), self.targets.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)));
        
        stats
    }
}

impl ModelMetrics {
    /// Calculate regression metrics
    pub fn calculate_regression_metrics(predictions: &[f64], targets: &[f64]) -> Self {
        let n = predictions.len() as f64;
        
        // Mean Squared Error
        let mse = predictions.iter()
            .zip(targets.iter())
            .map(|(pred, target)| (pred - target).powi(2))
            .sum::<f64>() / n;
        
        // Mean Absolute Error
        let mae = predictions.iter()
            .zip(targets.iter())
            .map(|(pred, target)| (pred - target).abs())
            .sum::<f64>() / n;
        
        Self {
            accuracy: 0.0, // Not applicable for regression
            precision: 0.0,
            recall: 0.0,
            f1_score: 0.0,
            mse,
            mae,
        }
    }
    
    /// Calculate classification metrics (simplified binary classification)
    pub fn calculate_classification_metrics(predictions: &[f64], targets: &[f64], threshold: f64) -> Self {
        let mut tp = 0.0;
        let mut fp = 0.0;
        let mut tn = 0.0;
        let mut fn_count = 0.0;
        
        for (pred, target) in predictions.iter().zip(targets.iter()) {
            let pred_class = if *pred >= threshold { 1.0 } else { 0.0 };
            let target_class = if *target >= threshold { 1.0 } else { 0.0 };
            
            match (pred_class, target_class) {
                (1.0, 1.0) => tp += 1.0,
                (1.0, 0.0) => fp += 1.0,
                (0.0, 0.0) => tn += 1.0,
                (0.0, 1.0) => fn_count += 1.0,
                _ => {}
            }
        }
        
        let accuracy = (tp + tn) / (tp + fp + tn + fn_count);
        let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
        let recall = if tp + fn_count > 0.0 { tp / (tp + fn_count) } else { 0.0 };
        let f1_score = if precision + recall > 0.0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 };
        
        Self {
            accuracy,
            precision,
            recall,
            f1_score,
            mse: 0.0,
            mae: 0.0,
        }
    }
}

/// Utility functions for ML operations
pub mod ml_utils {
    use super::*;
    
    /// Generate random training data for testing
    pub fn generate_random_data(samples: usize, features: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut feature_data = Vec::new();
        let mut targets = Vec::new();
        
        for _ in 0..samples {
            let mut sample = Vec::new();
            let mut target = 0.0;
            
            for _ in 0..features {
                let value = fastrand::f64() * 10.0 - 5.0; // Random value between -5 and 5
                sample.push(value);
                target += value * 0.5; // Simple linear relationship
            }
            
            target += fastrand::f64() * 0.1 - 0.05; // Add noise
            
            feature_data.push(sample);
            targets.push(target);
        }
        
        (feature_data, targets)
    }
    
    /// Cross-validation for model evaluation
    pub fn cross_validate<M: MLModel<Input = Vec<f64>, Output = f64> + Clone>(
        mut model: M,
        features: &[Vec<f64>],
        targets: &[f64],
        folds: usize,
    ) -> Result<Vec<f64>> {
        let fold_size = features.len() / folds;
        let mut scores = Vec::new();
        
        for fold in 0..folds {
            let start_idx = fold * fold_size;
            let end_idx = if fold == folds - 1 { features.len() } else { (fold + 1) * fold_size };
            
            // Split data
            let mut train_features = Vec::new();
            let mut train_targets = Vec::new();
            let mut test_features = Vec::new();
            let mut test_targets = Vec::new();
            
            for (i, (feature, target)) in features.iter().zip(targets.iter()).enumerate() {
                if i >= start_idx && i < end_idx {
                    test_features.push(feature.clone());
                    test_targets.push(*target);
                } else {
                    train_features.push(feature.clone());
                    train_targets.push(*target);
                }
            }
            
            // Train and evaluate
            model.train(&train_features, &train_targets)?;
            
            let mut predictions = Vec::new();
            for test_feature in &test_features {
                predictions.push(model.predict(test_feature)?);
            }
            
            let metrics = ModelMetrics::calculate_regression_metrics(&predictions, &test_targets);
            scores.push(1.0 / (1.0 + metrics.mse)); // Convert MSE to a score (higher is better)
        }
        
        Ok(scores)
    }
    
    /// Feature importance calculation (simplified)
    pub fn calculate_feature_importance(
        features: &[Vec<f64>],
        targets: &[f64],
        feature_names: &[String],
    ) -> Result<HashMap<String, f64>> {
        let mut importance = HashMap::new();
        
        // Calculate correlation-based importance
        for (feature_idx, feature_name) in feature_names.iter().enumerate() {
            let feature_values: Vec<f64> = features.iter().map(|f| f[feature_idx]).collect();
            let correlation = calculate_correlation(&feature_values, targets)?;
            importance.insert(feature_name.clone(), correlation.abs());
        }
        
        Ok(importance)
    }
    
    /// Calculate Pearson correlation coefficient
    fn calculate_correlation(x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != y.len() {
            return Err(CompilerError::InternalError("Arrays must have same length".to_string()));
        }
        
        let n = x.len() as f64;
        let mean_x = x.iter().sum::<f64>() / n;
        let mean_y = y.iter().sum::<f64>() / n;
        
        let mut numerator = 0.0;
        let mut sum_sq_x = 0.0;
        let mut sum_sq_y = 0.0;
        
        for (xi, yi) in x.iter().zip(y.iter()) {
            let dx = xi - mean_x;
            let dy = yi - mean_y;
            numerator += dx * dy;
            sum_sq_x += dx * dx;
            sum_sq_y += dy * dy;
        }
        
        let denominator = (sum_sq_x * sum_sq_y).sqrt();
        if denominator == 0.0 {
            Ok(0.0)
        } else {
            Ok(numerator / denominator)
        }
    }
}

// Export main types and functions
pub use ml_utils::*;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_neural_network_creation() {
        let nn = NeuralNetwork::new(&[3, 5, 2], 0.01, ActivationFunction::ReLU);
        assert_eq!(nn.layers.len(), 2);
        assert_eq!(nn.layers[0].neurons, 5);
        assert_eq!(nn.layers[1].neurons, 2);
    }
    
    #[test]
    fn test_linear_regression() {
        let mut lr = LinearRegression::new(2, 0.01);
        let inputs = vec![vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]];
        let targets = vec![3.0, 5.0, 7.0];
        
        assert!(lr.train(&inputs, &targets).is_ok());
        
        let prediction = lr.predict(&vec![4.0, 5.0]).unwrap();
        assert!(prediction > 0.0); // Should predict something reasonable
    }
    
    #[test]
    fn test_data_preprocessor() {
        let mut preprocessor = DataPreprocessor::new();
        let data = vec![
            vec![1.0, 2.0],
            vec![2.0, 4.0],
            vec![3.0, 6.0],
        ];
        
        assert!(preprocessor.fit(&data).is_ok());
        let transformed = preprocessor.transform(&data).unwrap();
        assert_eq!(transformed.len(), 3);
        assert_eq!(transformed[0].len(), 2);
    }
    
    #[test]
    fn test_ml_dataset() {
        let features = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let targets = vec![1.0, 2.0];
        let feature_names = vec!["feature1".to_string(), "feature2".to_string()];
        
        let dataset = MLDataset::new(features, targets, feature_names);
        let (train, test) = dataset.train_test_split(0.5);
        
        assert_eq!(train.features.len(), 1);
        assert_eq!(test.features.len(), 1);
    }
}