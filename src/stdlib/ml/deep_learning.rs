//! Advanced deep learning architectures and components
//! Includes LSTM, GRU, Transformers, attention mechanisms, and optimization algorithms

use crate::stdlib::ml::tensor::Tensor;
use crate::error::AugustiumError;

/// Activation functions
#[derive(Debug, Clone)]
pub enum ActivationFunction {
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    GELU,
    Swish,
    LeakyReLU(f32),
    ELU(f32),
}

/// Layer normalization
#[derive(Debug, Clone)]
pub struct LayerNorm {
    pub normalized_shape: Vec<usize>,
    pub eps: f32,
    pub weight: Tensor,
    pub bias: Tensor,
}

/// Batch normalization
#[derive(Debug, Clone)]
pub struct BatchNorm {
    pub num_features: usize,
    pub eps: f32,
    pub momentum: f32,
    pub weight: Tensor,
    pub bias: Tensor,
    pub running_mean: Tensor,
    pub running_var: Tensor,
    pub training: bool,
}

/// Dropout layer
#[derive(Debug, Clone)]
pub struct Dropout {
    pub p: f32,
    pub training: bool,
}

/// Linear/Dense layer
#[derive(Debug, Clone)]
pub struct Linear {
    pub in_features: usize,
    pub out_features: usize,
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

/// Convolutional layer
#[derive(Debug, Clone)]
pub struct Conv2d {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    pub dilation: (usize, usize),
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

/// LSTM cell
#[derive(Debug, Clone)]
pub struct LSTMCell {
    pub input_size: usize,
    pub hidden_size: usize,
    pub weight_ih: Tensor, // Input-to-hidden weights
    pub weight_hh: Tensor, // Hidden-to-hidden weights
    pub bias_ih: Option<Tensor>,
    pub bias_hh: Option<Tensor>,
}

/// LSTM layer
#[derive(Debug, Clone)]
pub struct LSTM {
    pub input_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub bias: bool,
    pub batch_first: bool,
    pub dropout: f32,
    pub bidirectional: bool,
    pub cells: Vec<LSTMCell>,
}

/// GRU cell
#[derive(Debug, Clone)]
pub struct GRUCell {
    pub input_size: usize,
    pub hidden_size: usize,
    pub weight_ih: Tensor,
    pub weight_hh: Tensor,
    pub bias_ih: Option<Tensor>,
    pub bias_hh: Option<Tensor>,
}

/// GRU layer
#[derive(Debug, Clone)]
pub struct GRU {
    pub input_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub bias: bool,
    pub batch_first: bool,
    pub dropout: f32,
    pub bidirectional: bool,
    pub cells: Vec<GRUCell>,
}

/// Multi-head attention
#[derive(Debug, Clone)]
pub struct MultiHeadAttention {
    pub embed_dim: usize,
    pub num_heads: usize,
    pub dropout: f32,
    pub head_dim: usize,
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,
}

/// Transformer encoder layer
#[derive(Debug, Clone)]
pub struct TransformerEncoderLayer {
    pub d_model: usize,
    pub nhead: usize,
    pub dim_feedforward: usize,
    pub dropout: f32,
    pub self_attn: MultiHeadAttention,
    pub linear1: Linear,
    pub linear2: Linear,
    pub norm1: LayerNorm,
    pub norm2: LayerNorm,
    pub dropout1: Dropout,
    pub dropout2: Dropout,
    pub activation: ActivationFunction,
}

/// Transformer decoder layer
#[derive(Debug, Clone)]
pub struct TransformerDecoderLayer {
    pub d_model: usize,
    pub nhead: usize,
    pub dim_feedforward: usize,
    pub dropout: f32,
    pub self_attn: MultiHeadAttention,
    pub multihead_attn: MultiHeadAttention,
    pub linear1: Linear,
    pub linear2: Linear,
    pub norm1: LayerNorm,
    pub norm2: LayerNorm,
    pub norm3: LayerNorm,
    pub dropout1: Dropout,
    pub dropout2: Dropout,
    pub dropout3: Dropout,
    pub activation: ActivationFunction,
}

/// Positional encoding for transformers
#[derive(Debug, Clone)]
pub struct PositionalEncoding {
    pub d_model: usize,
    pub max_len: usize,
    pub encoding: Tensor,
}

/// Optimizer trait
pub trait Optimizer {
    fn step(&mut self, parameters: &mut [Tensor]) -> Result<(), AugustiumError>;
    fn zero_grad(&mut self, parameters: &mut [Tensor]);
}

/// Adam optimizer
#[derive(Debug, Clone)]
pub struct Adam {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub step_count: usize,
    pub m: Vec<Tensor>, // First moment estimates
    pub v: Vec<Tensor>, // Second moment estimates
}

/// SGD optimizer
#[derive(Debug, Clone)]
pub struct SGD {
    pub lr: f32,
    pub momentum: f32,
    pub weight_decay: f32,
    pub dampening: f32,
    pub nesterov: bool,
    pub velocity: Vec<Tensor>,
}

/// RMSprop optimizer
#[derive(Debug, Clone)]
pub struct RMSprop {
    pub lr: f32,
    pub alpha: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub momentum: f32,
    pub centered: bool,
    pub square_avg: Vec<Tensor>,
    pub momentum_buffer: Vec<Tensor>,
    pub grad_avg: Vec<Tensor>,
}

/// Implementation of activation functions
impl ActivationFunction {
    pub fn apply(&self, input: &Tensor) -> Result<Tensor, AugustiumError> {
        match self {
            ActivationFunction::ReLU => {
                input.map(|x| x.max(0.0))
            },
            ActivationFunction::Sigmoid => {
                input.map(|x| 1.0 / (1.0 + (-x).exp()))
            },
            ActivationFunction::Tanh => {
                input.map(|x| x.tanh())
            },
            ActivationFunction::Softmax => {
                let exp_input = input.exp()?;
                let sum = exp_input.sum(Some(input.ndim() - 1), true)?;
                exp_input.div(&sum)
            },
            ActivationFunction::GELU => {
                input.map(|x| 0.5 * x * (1.0 + (x * 0.7978845608 * (1.0 + 0.044715 * x * x)).tanh()))
            },
            ActivationFunction::Swish => {
                let sigmoid = input.map(|x| 1.0 / (1.0 + (-x).exp()))?;
                input.mul(&sigmoid)
            },
            ActivationFunction::LeakyReLU(alpha) => {
                input.map(|x| if x > 0.0 { x } else { alpha * x })
            },
            ActivationFunction::ELU(alpha) => {
                input.map(|x| if x > 0.0 { x } else { alpha * (x.exp() - 1.0) })
            },
        }
    }
    
    pub fn derivative(&self, input: &Tensor) -> Result<Tensor, AugustiumError> {
        match self {
            ActivationFunction::ReLU => {
                input.map(|x| if x > 0.0 { 1.0 } else { 0.0 })
            },
            ActivationFunction::Sigmoid => {
                let sigmoid = self.apply(input)?;
                let one = Tensor::ones(sigmoid.shape().dims.clone())?;
                let one_minus_sigmoid = one.sub(&sigmoid)?;
                sigmoid.mul(&one_minus_sigmoid)
            },
            ActivationFunction::Tanh => {
                let tanh_val = input.map(|x| x.tanh())?;
                let one = Tensor::ones(tanh_val.shape().dims.clone())?;
                let tanh_squared = tanh_val.mul(&tanh_val)?;
                one.sub(&tanh_squared)
            },
            _ => {
                // Simplified derivatives for other functions
                Tensor::ones(input.shape().dims.clone())
            },
        }
    }
}

/// Layer normalization implementation
impl LayerNorm {
    pub fn new(normalized_shape: Vec<usize>, eps: f32) -> Result<Self, AugustiumError> {
        let weight = Tensor::ones(normalized_shape.clone())?;
        let bias = Tensor::zeros(normalized_shape.clone())?;
        
        Ok(LayerNorm {
            normalized_shape,
            eps,
            weight,
            bias,
        })
    }
    
    pub fn forward(&self, input: &Tensor) -> Result<Tensor, AugustiumError> {
        // Calculate mean and variance along the last dimensions
        let mean = input.mean(Some(input.ndim() - 1), true)?;
        let centered = input.sub(&mean)?;
        let variance = centered.mul(&centered)?.mean(Some(input.ndim() - 1), true)?;
        let std = variance.add_scalar(self.eps)?.sqrt()?;
        
        let normalized = centered.div(&std)?;
        let scaled = normalized.mul(&self.weight)?;
        scaled.add(&self.bias)
    }
}

/// Batch normalization implementation
impl BatchNorm {
    pub fn new(num_features: usize, eps: f32, momentum: f32) -> Result<Self, AugustiumError> {
        let weight = Tensor::ones(vec![num_features])?;
        let bias = Tensor::zeros(vec![num_features])?;
        let running_mean = Tensor::zeros(vec![num_features])?;
        let running_var = Tensor::ones(vec![num_features])?;
        
        Ok(BatchNorm {
            num_features,
            eps,
            momentum,
            weight,
            bias,
            running_mean,
            running_var,
            training: true,
        })
    }
    
    pub fn forward(&mut self, input: &Tensor) -> Result<Tensor, AugustiumError> {
        if self.training {
            // Calculate batch statistics
            let mean = input.mean(Some(0), true)?;
            let variance = input.sub(&mean)?.pow(2.0)?.mean(Some(0), true)?;
            
            // Update running statistics
            let momentum_complement = 1.0 - self.momentum;
            self.running_mean = self.running_mean.mul_scalar(momentum_complement)?
                .add(&mean.mul_scalar(self.momentum)?)?;
            self.running_var = self.running_var.mul_scalar(momentum_complement)?
                .add(&variance.mul_scalar(self.momentum)?)?;
            
            // Normalize
            let normalized = input.sub(&mean)?
                .div(&variance.add_scalar(self.eps)?.sqrt()?)?;
            
            normalized.mul(&self.weight)?.add(&self.bias)
        } else {
            // Use running statistics
            let normalized = input.sub(&self.running_mean)?
                .div(&self.running_var.add_scalar(self.eps)?.sqrt()?)?;
            
            normalized.mul(&self.weight)?.add(&self.bias)
        }
    }
    
    pub fn eval(&mut self) {
        self.training = false;
    }
    
    pub fn train(&mut self) {
        self.training = true;
    }
}

/// Dropout implementation
impl Dropout {
    pub fn new(p: f32) -> Self {
        Dropout { p, training: true }
    }
    
    pub fn forward(&self, input: &Tensor) -> Result<Tensor, AugustiumError> {
        if self.training && self.p > 0.0 {
            let keep_prob = 1.0 - self.p;
            let mask = Tensor::rand(input.shape().dims.clone(), 0.0, 1.0)?;
            let binary_mask = mask.map(|x| if x < keep_prob { 1.0 / keep_prob } else { 0.0 })?;
            input.mul(&binary_mask)
        } else {
            Ok(input.clone())
        }
    }
    
    pub fn eval(&mut self) {
        self.training = false;
    }
    
    pub fn train(&mut self) {
        self.training = true;
    }
}

/// Linear layer implementation
impl Linear {
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Result<Self, AugustiumError> {
        // Xavier/Glorot initialization
        let bound = (6.0 / (in_features + out_features) as f32).sqrt();
        let weight = Tensor::rand(vec![out_features, in_features], -bound, bound)?;
        let bias_tensor = if bias {
            Some(Tensor::rand(vec![out_features], -bound, bound)?)
        } else {
            None
        };
        
        Ok(Linear {
            in_features,
            out_features,
            weight,
            bias: bias_tensor,
        })
    }
    
    pub fn forward(&self, input: &Tensor) -> Result<Tensor, AugustiumError> {
        let output = input.matmul(&self.weight.transpose()?)?;
        if let Some(ref bias) = self.bias {
            output.add(bias)
        } else {
            Ok(output)
        }
    }
}

/// Conv2d layer implementation
impl Conv2d {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize, stride: usize, padding: usize) -> Result<Self, AugustiumError> {
        let weight = Tensor::rand(vec![out_channels, in_channels, kernel_size, kernel_size], -0.1, 0.1)?;
        let bias = Some(Tensor::zeros(vec![out_channels])?);
        
        Ok(Conv2d {
            in_channels,
            out_channels,
            kernel_size: (kernel_size, kernel_size),
            stride: (stride, stride),
            padding: (padding, padding),
            dilation: (1, 1),
            weight,
            bias,
        })
    }
    
    pub fn forward(&self, input: &Tensor) -> Result<Tensor, AugustiumError> {
        // Simplified convolution - just return input for now
        Ok(input.clone())
    }
}

/// LSTM layer implementation
impl LSTM {
    pub fn new(input_size: usize, hidden_size: usize) -> Result<Self, AugustiumError> {
        let cell = LSTMCell::new(input_size, hidden_size, true)?;
        
        Ok(LSTM {
            input_size,
            hidden_size,
            num_layers: 1,
            bias: true,
            batch_first: false,
            dropout: 0.0,
            bidirectional: false,
            cells: vec![cell],
        })
    }
    
    pub fn forward(&self, input: &Tensor) -> Result<Tensor, AugustiumError> {
        // Simplified LSTM forward - just return input for now
        Ok(input.clone())
    }
}

/// LSTM cell implementation
impl LSTMCell {
    pub fn new(input_size: usize, hidden_size: usize, bias: bool) -> Result<Self, AugustiumError> {
        let bound = (1.0 / hidden_size as f32).sqrt();
        
        // Weight matrices for input-to-hidden (4 gates: i, f, g, o)
        let weight_ih = Tensor::rand(vec![4 * hidden_size, input_size], -bound, bound)?;
        
        // Weight matrices for hidden-to-hidden (4 gates: i, f, g, o)
        let weight_hh = Tensor::rand(vec![4 * hidden_size, hidden_size], -bound, bound)?;
        
        let bias_ih = if bias {
            Some(Tensor::rand(vec![4 * hidden_size], -bound, bound)?)
        } else {
            None
        };
        
        let bias_hh = if bias {
            Some(Tensor::rand(vec![4 * hidden_size], -bound, bound)?)
        } else {
            None
        };
        
        Ok(LSTMCell {
            input_size,
            hidden_size,
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
        })
    }
    
    pub fn forward(&self, input: &Tensor, hidden: &Tensor, cell: &Tensor) -> Result<(Tensor, Tensor), AugustiumError> {
        // Compute gates
        let gi = input.matmul(&self.weight_ih.transpose()?)?;
        let gh = hidden.matmul(&self.weight_hh.transpose()?)?;
        let mut gates = gi.add(&gh)?;
        
        if let Some(ref bias_ih) = self.bias_ih {
            gates = gates.add(bias_ih)?;
        }
        if let Some(ref bias_hh) = self.bias_hh {
            gates = gates.add(bias_hh)?;
        }
        
        // Split gates into input, forget, cell, output
        let gate_size = self.hidden_size;
        let input_gate = self.slice_tensor(&gates, 0, gate_size)?;
        let forget_gate = self.slice_tensor(&gates, gate_size, 2 * gate_size)?;
        let cell_gate = self.slice_tensor(&gates, 2 * gate_size, 3 * gate_size)?;
        let output_gate = self.slice_tensor(&gates, 3 * gate_size, 4 * gate_size)?;
        
        // Apply activations
        let i = ActivationFunction::Sigmoid.apply(&input_gate)?;
        let f = ActivationFunction::Sigmoid.apply(&forget_gate)?;
        let g = ActivationFunction::Tanh.apply(&cell_gate)?;
        let o = ActivationFunction::Sigmoid.apply(&output_gate)?;
        
        // Update cell state
        let new_cell = f.mul(cell)?.add(&i.mul(&g)?)?;
        
        // Update hidden state
        let new_hidden = o.mul(&ActivationFunction::Tanh.apply(&new_cell)?)?;
        
        Ok((new_hidden, new_cell))
    }
    
    fn slice_tensor(&self, tensor: &Tensor, start: usize, end: usize) -> Result<Tensor, AugustiumError> {
        // Simplified tensor slicing - would need proper implementation
        let data = tensor.to_vec();
        let sliced_data = data[start..end].to_vec();
        Tensor::from_data(sliced_data, vec![end - start])
    }
}

/// Multi-head attention implementation
impl MultiHeadAttention {
    pub fn new(embed_dim: usize, num_heads: usize, dropout: f32) -> Result<Self, AugustiumError> {
        if embed_dim % num_heads != 0 {
            return Err(AugustiumError::Runtime(
                "embed_dim must be divisible by num_heads".to_string()
            ));
        }
        
        let head_dim = embed_dim / num_heads;
        
        let q_proj = Linear::new(embed_dim, embed_dim, true)?;
        let k_proj = Linear::new(embed_dim, embed_dim, true)?;
        let v_proj = Linear::new(embed_dim, embed_dim, true)?;
        let out_proj = Linear::new(embed_dim, embed_dim, true)?;
        
        Ok(MultiHeadAttention {
            embed_dim,
            num_heads,
            dropout,
            head_dim,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
        })
    }
    
    pub fn forward(&self, query: &Tensor, key: &Tensor, value: &Tensor, 
                   attn_mask: Option<&Tensor>) -> Result<Tensor, AugustiumError> {
        let batch_size = query.shape().dims[0];
        let seq_len = query.shape().dims[1];
        
        // Project to Q, K, V
        let q = self.q_proj.forward(query)?;
        let k = self.k_proj.forward(key)?;
        let v = self.v_proj.forward(value)?;
        
        // Reshape for multi-head attention
        let q = q.reshape(vec![batch_size, seq_len, self.num_heads, self.head_dim])?
            .permute(vec![0, 2, 1, 3])?; // [batch, heads, seq_len, head_dim]
        let k = k.reshape(vec![batch_size, seq_len, self.num_heads, self.head_dim])?
            .permute(vec![0, 2, 1, 3])?;
        let v = v.reshape(vec![batch_size, seq_len, self.num_heads, self.head_dim])?
            .permute(vec![0, 2, 1, 3])?;
        
        // Scaled dot-product attention
        let scale = (self.head_dim as f32).sqrt();
        let scores = q.matmul(&k.transpose()?)?.div_scalar(scale)?;
        
        // Apply attention mask if provided
        let scores = if let Some(mask) = attn_mask {
            scores.add(mask)?
        } else {
            scores
        };
        
        // Apply softmax
        let attn_weights = ActivationFunction::Softmax.apply(&scores)?;
        
        // Apply attention to values
        let attn_output = attn_weights.matmul(&v)?;
        
        // Reshape back
        let attn_output = attn_output.permute(vec![0, 2, 1, 3])?
            .reshape(vec![batch_size, seq_len, self.embed_dim])?;
        
        // Final projection
        self.out_proj.forward(&attn_output)
    }
}

/// Transformer encoder layer implementation
impl TransformerEncoderLayer {
    pub fn new(d_model: usize, nhead: usize, dim_feedforward: usize, dropout: f32) -> Result<Self, AugustiumError> {
        let self_attn = MultiHeadAttention::new(d_model, nhead, dropout)?;
        let linear1 = Linear::new(d_model, dim_feedforward, true)?;
        let linear2 = Linear::new(dim_feedforward, d_model, true)?;
        let norm1 = LayerNorm::new(vec![d_model], 1e-5)?;
        let norm2 = LayerNorm::new(vec![d_model], 1e-5)?;
        let dropout1 = Dropout::new(dropout);
        let dropout2 = Dropout::new(dropout);
        let activation = ActivationFunction::ReLU;
        
        Ok(TransformerEncoderLayer {
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            self_attn,
            linear1,
            linear2,
            norm1,
            norm2,
            dropout1,
            dropout2,
            activation,
        })
    }
    
    pub fn forward(&self, src: &Tensor, src_mask: Option<&Tensor>) -> Result<Tensor, AugustiumError> {
        // Self-attention
        let attn_output = self.self_attn.forward(src, src, src, src_mask)?;
        let src2 = self.dropout1.forward(&attn_output)?;
        let src = src.add(&src2)?;
        let src = self.norm1.forward(&src)?;
        
        // Feed forward
        let ff_output = self.linear1.forward(&src)?;
        let ff_output = self.activation.apply(&ff_output)?;
        let ff_output = self.dropout2.forward(&ff_output)?;
        let ff_output = self.linear2.forward(&ff_output)?;
        
        let src2 = src.add(&ff_output)?;
        self.norm2.forward(&src2)
    }
}

/// Positional encoding implementation
impl PositionalEncoding {
    pub fn new(d_model: usize, max_len: usize) -> Result<Self, AugustiumError> {
        let mut encoding_data = vec![0.0; max_len * d_model];
        
        for pos in 0..max_len {
            for i in (0..d_model).step_by(2) {
                let angle = pos as f32 / 10000.0_f32.powf(i as f32 / d_model as f32);
                encoding_data[pos * d_model + i] = angle.sin();
                if i + 1 < d_model {
                    encoding_data[pos * d_model + i + 1] = angle.cos();
                }
            }
        }
        
        let encoding = Tensor::from_data(encoding_data, vec![max_len, d_model])?;
        
        Ok(PositionalEncoding {
            d_model,
            max_len,
            encoding,
        })
    }
    
    pub fn forward(&self, input: &Tensor) -> Result<Tensor, AugustiumError> {
        let seq_len = input.shape().dims[1];
        if seq_len > self.max_len {
            return Err(AugustiumError::Runtime(
                format!("Sequence length {} exceeds maximum {}", seq_len, self.max_len)
            ));
        }
        
        // Slice positional encoding to match input sequence length
        let pos_encoding = self.slice_encoding(seq_len)?;
        input.add(&pos_encoding)
    }
    
    fn slice_encoding(&self, seq_len: usize) -> Result<Tensor, AugustiumError> {
        // Simplified slicing - would need proper implementation
        let data = self.encoding.to_vec();
        let sliced_data = data[..seq_len * self.d_model].to_vec();
        Tensor::from_data(sliced_data, vec![seq_len, self.d_model])
    }
}

/// Adam optimizer implementation
impl Adam {
    pub fn new(lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) -> Self {
        Adam {
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            step_count: 0,
            m: Vec::new(),
            v: Vec::new(),
        }
    }
    
    pub fn init_state(&mut self, parameters: &[Tensor]) -> Result<(), AugustiumError> {
        self.m.clear();
        self.v.clear();
        
        for param in parameters {
            self.m.push(Tensor::zeros(param.shape().dims.clone())?);
            self.v.push(Tensor::zeros(param.shape().dims.clone())?);
        }
        
        Ok(())
    }
}

impl Optimizer for Adam {
    fn step(&mut self, parameters: &mut [Tensor]) -> Result<(), AugustiumError> {
        if self.m.is_empty() {
            self.init_state(parameters)?;
        }
        
        self.step_count += 1;
        let bias_correction1 = 1.0 - self.beta1.powi(self.step_count as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.step_count as i32);
        
        for (i, param) in parameters.iter_mut().enumerate() {
            if let Some(grad) = param.grad() {
                let mut grad = grad.clone();
                
                // Add weight decay
                if self.weight_decay != 0.0 {
                    grad = grad.add(&param.mul_scalar(self.weight_decay)?)?;
                }
                
                // Update biased first moment estimate
                self.m[i] = self.m[i].mul_scalar(self.beta1)?
                    .add(&grad.mul_scalar(1.0 - self.beta1)?)?;
                
                // Update biased second raw moment estimate
                let grad_squared = grad.mul(&grad)?;
                self.v[i] = self.v[i].mul_scalar(self.beta2)?
                    .add(&grad_squared.mul_scalar(1.0 - self.beta2)?)?;
                
                // Compute bias-corrected first moment estimate
                let m_hat = self.m[i].div_scalar(bias_correction1)?;
                
                // Compute bias-corrected second raw moment estimate
                let v_hat = self.v[i].div_scalar(bias_correction2)?;
                
                // Update parameters
                let denominator = v_hat.sqrt()?.add_scalar(self.eps)?;
                let update = m_hat.div(&denominator)?.mul_scalar(self.lr)?;
                *param = param.sub(&update)?;
            }
        }
        
        Ok(())
    }
    
    fn zero_grad(&mut self, parameters: &mut [Tensor]) {
        for param in parameters {
            param.zero_grad();
        }
    }
}

/// SGD optimizer implementation
impl SGD {
    pub fn new(lr: f32, momentum: f32, weight_decay: f32, dampening: f32, nesterov: bool) -> Self {
        SGD {
            lr,
            momentum,
            weight_decay,
            dampening,
            nesterov,
            velocity: Vec::new(),
        }
    }
    
    pub fn init_state(&mut self, parameters: &[Tensor]) -> Result<(), AugustiumError> {
        self.velocity.clear();
        
        for param in parameters {
            self.velocity.push(Tensor::zeros(param.shape().dims.clone())?);
        }
        
        Ok(())
    }
}

impl Optimizer for SGD {
    fn step(&mut self, parameters: &mut [Tensor]) -> Result<(), AugustiumError> {
        if self.velocity.is_empty() {
            self.init_state(parameters)?;
        }
        
        for (i, param) in parameters.iter_mut().enumerate() {
            if let Some(grad) = param.grad() {
                let mut grad = grad.clone();
                
                // Add weight decay
                if self.weight_decay != 0.0 {
                    grad = grad.add(&param.mul_scalar(self.weight_decay)?)?;
                }
                
                if self.momentum != 0.0 {
                    // Update velocity
                    self.velocity[i] = self.velocity[i].mul_scalar(self.momentum)?
                        .add(&grad.mul_scalar(1.0 - self.dampening)?)?;
                    
                    if self.nesterov {
                        grad = grad.add(&self.velocity[i].mul_scalar(self.momentum)?)?;
                    } else {
                        grad = self.velocity[i].clone();
                    }
                }
                
                // Update parameters
                let update = grad.mul_scalar(self.lr)?;
                *param = param.sub(&update)?;
            }
        }
        
        Ok(())
    }
    
    fn zero_grad(&mut self, parameters: &mut [Tensor]) {
        for param in parameters {
            param.zero_grad();
        }
    }
}

/// Learning rate scheduler trait
pub trait LRScheduler {
    fn step(&mut self) -> f32;
    fn get_lr(&self) -> f32;
}

/// Step learning rate scheduler
#[derive(Debug, Clone)]
pub struct StepLR {
    pub initial_lr: f32,
    pub step_size: usize,
    pub gamma: f32,
    pub current_step: usize,
}

impl StepLR {
    pub fn new(initial_lr: f32, step_size: usize, gamma: f32) -> Self {
        StepLR {
            initial_lr,
            step_size,
            gamma,
            current_step: 0,
        }
    }
}

impl LRScheduler for StepLR {
    fn step(&mut self) -> f32 {
        self.current_step += 1;
        self.get_lr()
    }
    
    fn get_lr(&self) -> f32 {
        let decay_factor = self.gamma.powi((self.current_step / self.step_size) as i32);
        self.initial_lr * decay_factor
    }
}

/// Exponential learning rate scheduler
#[derive(Debug, Clone)]
pub struct ExponentialLR {
    pub initial_lr: f32,
    pub gamma: f32,
    pub current_step: usize,
}

impl ExponentialLR {
    pub fn new(initial_lr: f32, gamma: f32) -> Self {
        ExponentialLR {
            initial_lr,
            gamma,
            current_step: 0,
        }
    }
}

impl LRScheduler for ExponentialLR {
    fn step(&mut self) -> f32 {
        self.current_step += 1;
        self.get_lr()
    }
    
    fn get_lr(&self) -> f32 {
        self.initial_lr * self.gamma.powi(self.current_step as i32)
    }
}

/// Cosine annealing learning rate scheduler
#[derive(Debug, Clone)]
pub struct CosineAnnealingLR {
    pub initial_lr: f32,
    pub min_lr: f32,
    pub t_max: usize,
    pub current_step: usize,
}

impl CosineAnnealingLR {
    pub fn new(initial_lr: f32, t_max: usize, min_lr: f32) -> Self {
        CosineAnnealingLR {
            initial_lr,
            min_lr,
            t_max,
            current_step: 0,
        }
    }
}

impl LRScheduler for CosineAnnealingLR {
    fn step(&mut self) -> f32 {
        self.current_step += 1;
        self.get_lr()
    }
    
    fn get_lr(&self) -> f32 {
        let progress = (self.current_step % self.t_max) as f32 / self.t_max as f32;
        let cosine_factor = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
        self.min_lr + (self.initial_lr - self.min_lr) * cosine_factor
    }
}