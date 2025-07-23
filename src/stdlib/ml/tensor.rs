//! Advanced tensor operations for ML computations
//! Provides comprehensive tensor arithmetic, linear algebra, and specialized ML operations

#[cfg(feature = "ml-basic")]
mod tensor_impl {
    use std::fmt;
    use ndarray::{ArrayD, IxDyn, Axis};
    use crate::error::AugustiumError;

    /// Data types supported by tensors
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum DataType {
        Float32,
        Float64,
        Int32,
        Int64,
        Bool,
        Complex64,
        Complex128,
    }

    /// Tensor shape representation
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct TensorShape {
        pub dims: Vec<usize>,
    }

    /// Advanced tensor with support for various data types and operations
    #[derive(Debug, Clone)]
    pub struct Tensor {
        pub data: ArrayD<f32>, // Primary data storage
        pub shape: TensorShape,
        pub dtype: DataType,
        pub requires_grad: bool,
        pub grad: Option<Box<Tensor>>,
        pub grad_fn: Option<String>, // For backpropagation
    }

    /// Tensor creation and initialization
    impl Tensor {
        /// Create a new tensor with given shape and data type
        pub fn new(shape: Vec<usize>, dtype: DataType) -> Result<Self, AugustiumError> {
            let data = ArrayD::zeros(IxDyn(&shape));
            
            Ok(Tensor {
                data,
                shape: TensorShape { dims: shape },
                dtype,
                requires_grad: false,
                grad: None,
                grad_fn: None,
            })
        }

        /// Create tensor from raw data
        pub fn from_data(data: Vec<f32>, shape: Vec<usize>) -> Result<Self, AugustiumError> {
            let total_elements: usize = shape.iter().product();
            if data.len() != total_elements {
                return Err(AugustiumError::Runtime(
                    "Data length doesn't match tensor shape".to_string()
                ));
            }
            
            let array_data = ArrayD::from_shape_vec(IxDyn(&shape), data)
                .map_err(|e| AugustiumError::Runtime(format!("Shape error: {}", e)))?;
            
            Ok(Tensor {
                data: array_data,
                shape: TensorShape { dims: shape },
                dtype: DataType::Float32,
                requires_grad: false,
                grad: None,
                grad_fn: None,
            })
        }

        /// Create tensor filled with zeros
        pub fn zeros(shape: Vec<usize>) -> Result<Self, AugustiumError> {
            Self::new(shape, DataType::Float32)
        }

        /// Create tensor filled with ones
        pub fn ones(shape: Vec<usize>) -> Result<Self, AugustiumError> {
            let mut tensor = Self::new(shape, DataType::Float32)?;
            tensor.data.fill(1.0);
            Ok(tensor)
        }

        /// Basic tensor operations
        pub fn shape(&self) -> &TensorShape {
            &self.shape
        }

        pub fn ndim(&self) -> usize {
            self.shape.dims.len()
        }

        pub fn numel(&self) -> usize {
            self.shape.total_elements()
        }

        /// Arithmetic operations
        pub fn add(&self, other: &Tensor) -> Result<Tensor, AugustiumError> {
            if !self.shape.is_broadcastable(&other.shape) {
                return Err(AugustiumError::Runtime(
                    "Tensors are not broadcastable for addition".to_string()
                ));
            }
            
            let result_data = &self.data + &other.data;
            Ok(Tensor {
                data: result_data,
                shape: self.shape.clone(),
                dtype: self.dtype,
                requires_grad: self.requires_grad || other.requires_grad,
                grad: None,
                grad_fn: Some("AddBackward".to_string()),
            })
        }

        pub fn mul(&self, other: &Tensor) -> Result<Tensor, AugustiumError> {
            if !self.shape.is_broadcastable(&other.shape) {
                return Err(AugustiumError::Runtime(
                    "Tensors are not broadcastable for multiplication".to_string()
                ));
            }
            
            let result_data = &self.data * &other.data;
            Ok(Tensor {
                data: result_data,
                shape: self.shape.clone(),
                dtype: self.dtype,
                requires_grad: self.requires_grad || other.requires_grad,
                grad: None,
                grad_fn: Some("MulBackward".to_string()),
            })
        }

        pub fn sub(&self, other: &Tensor) -> Result<Tensor, AugustiumError> {
            if !self.shape.is_broadcastable(&other.shape) {
                return Err(AugustiumError::Runtime(
                    "Tensors are not broadcastable for subtraction".to_string()
                ));
            }
            
            let result_data = &self.data - &other.data;
            Ok(Tensor {
                data: result_data,
                shape: self.shape.clone(),
                dtype: self.dtype,
                requires_grad: self.requires_grad || other.requires_grad,
                grad: None,
                grad_fn: Some("SubBackward".to_string()),
            })
        }

        pub fn mul_scalar(&self, scalar: f32) -> Result<Tensor, AugustiumError> {
            let result_data = &self.data * scalar;
            Ok(Tensor {
                data: result_data,
                shape: self.shape.clone(),
                dtype: self.dtype,
                requires_grad: self.requires_grad,
                grad: None,
                grad_fn: Some("MulScalarBackward".to_string()),
            })
        }

        pub fn pow(&self, exponent: f32) -> Result<Tensor, AugustiumError> {
            let result_data = self.data.mapv(|x| x.powf(exponent));
            Ok(Tensor {
                data: result_data,
                shape: self.shape.clone(),
                dtype: self.dtype,
                requires_grad: self.requires_grad,
                grad: None,
                grad_fn: Some("PowBackward".to_string()),
            })
        }

        pub fn mean(&self, axis: Option<usize>, keepdims: bool) -> Result<Tensor, AugustiumError> {
            let result_data = if let Some(ax) = axis {
                let mean_data = self.data.mean_axis(Axis(ax)).unwrap();
                if keepdims {
                    let mut new_shape = self.shape.dims.clone();
                    new_shape[ax] = 1;
                    mean_data.into_shape(new_shape).unwrap()
                } else {
                    mean_data
                }
            } else {
                let mean_val = self.data.mean().unwrap();
                ArrayD::from_elem(vec![1], mean_val)
            };
            
            let shape_vec = result_data.shape().to_vec();
            Ok(Tensor {
                data: result_data,
                shape: TensorShape::new(shape_vec),
                dtype: self.dtype,
                requires_grad: self.requires_grad,
                grad: None,
                grad_fn: Some("MeanBackward".to_string()),
            })
        }

        pub fn add_scalar(&self, scalar: f32) -> Result<Tensor, AugustiumError> {
            let result_data = &self.data + scalar;
            Ok(Tensor {
                data: result_data,
                shape: self.shape.clone(),
                dtype: self.dtype,
                requires_grad: self.requires_grad,
                grad: None,
                grad_fn: Some("AddScalarBackward".to_string()),
            })
        }

        pub fn div(&self, other: &Tensor) -> Result<Tensor, AugustiumError> {
            if !self.shape.is_broadcastable(&other.shape) {
                return Err(AugustiumError::Runtime(
                    "Tensors are not broadcastable for division".to_string()
                ));
            }
            
            let result_data = &self.data / &other.data;
            Ok(Tensor {
                data: result_data,
                shape: self.shape.clone(),
                dtype: self.dtype,
                requires_grad: self.requires_grad || other.requires_grad,
                grad: None,
                grad_fn: Some("DivBackward".to_string()),
            })
        }

        pub fn sqrt(&self) -> Result<Tensor, AugustiumError> {
            let result_data = self.data.mapv(|x| x.sqrt());
            Ok(Tensor {
                data: result_data,
                shape: self.shape.clone(),
                dtype: self.dtype,
                requires_grad: self.requires_grad,
                grad: None,
                grad_fn: Some("SqrtBackward".to_string()),
            })
        }

        pub fn map<F>(&self, f: F) -> Result<Tensor, AugustiumError>
        where
            F: Fn(f32) -> f32,
        {
            let result_data = self.data.mapv(f);
            Ok(Tensor {
                data: result_data,
                shape: self.shape.clone(),
                dtype: self.dtype,
                requires_grad: self.requires_grad,
                grad: None,
                grad_fn: Some("MapBackward".to_string()),
            })
        }

        pub fn exp(&self) -> Result<Tensor, AugustiumError> {
            let result_data = self.data.mapv(|x| x.exp());
            Ok(Tensor {
                data: result_data,
                shape: self.shape.clone(),
                dtype: self.dtype,
                requires_grad: self.requires_grad,
                grad: None,
                grad_fn: Some("ExpBackward".to_string()),
            })
        }

        pub fn rand(shape: Vec<usize>, min: f32, max: f32) -> Result<Tensor, AugustiumError> {
            use rand::Rng;
            let total_elements: usize = shape.iter().product();
            let mut rng = rand::thread_rng();
            let data: Vec<f32> = (0..total_elements)
                .map(|_| rng.gen_range(min..max))
                .collect();
            
            let array_data = ArrayD::from_shape_vec(IxDyn(&shape), data)
                .map_err(|e| AugustiumError::Runtime(format!("Failed to create tensor from random data: {}", e)))?;
            
            Ok(Tensor {
                data: array_data,
                shape: TensorShape::new(shape),
                dtype: DataType::Float32,
                requires_grad: false,
                grad: None,
                grad_fn: Some("RandBackward".to_string()),
            })
        }

        pub fn sum(&self, axis: Option<usize>, keepdims: bool) -> Result<Tensor, AugustiumError> {
            let result_data = if let Some(ax) = axis {
                let sum_data = self.data.sum_axis(Axis(ax));
                if keepdims {
                    let mut new_shape = self.shape.dims.clone();
                    new_shape[ax] = 1;
                    sum_data.into_shape(new_shape).unwrap()
                } else {
                    sum_data
                }
            } else {
                let sum_val = self.data.sum();
                ArrayD::from_elem(vec![1], sum_val)
            };
            
            let shape_vec = result_data.shape().to_vec();
            Ok(Tensor {
                data: result_data,
                shape: TensorShape::new(shape_vec),
                dtype: self.dtype,
                requires_grad: self.requires_grad,
                grad: None,
                grad_fn: Some("SumBackward".to_string()),
            })
        }

        pub fn matmul(&self, other: &Tensor) -> Result<Tensor, AugustiumError> {
            // Simplified matrix multiplication - assumes 2D tensors
            if self.shape.dims.len() != 2 || other.shape.dims.len() != 2 {
                return Err(AugustiumError::Runtime(
                    "Matrix multiplication requires 2D tensors".to_string()
                ));
            }
            
            if self.shape.dims[1] != other.shape.dims[0] {
                return Err(AugustiumError::Runtime(
                    "Incompatible dimensions for matrix multiplication".to_string()
                ));
            }
            
            // Manual matrix multiplication to avoid ndarray dot issues
            let rows = self.shape.dims[0];
            let cols = other.shape.dims[1];
            let inner = self.shape.dims[1];
            
            let mut result_vec = vec![0.0; rows * cols];
            for i in 0..rows {
                for j in 0..cols {
                    let mut sum = 0.0;
                    for k in 0..inner {
                        let a_val = self.data[[i, k]];
                        let b_val = other.data[[k, j]];
                        sum += a_val * b_val;
                    }
                    result_vec[i * cols + j] = sum;
                }
            }
            
            let result_data = ArrayD::from_shape_vec(vec![rows, cols], result_vec)
                .map_err(|e| AugustiumError::Runtime(format!("Failed to create result tensor: {}", e)))?;
            
            Ok(Tensor {
                data: result_data,
                shape: TensorShape::new(vec![rows, cols]),
                dtype: self.dtype,
                requires_grad: self.requires_grad || other.requires_grad,
                grad: None,
                grad_fn: Some("MatmulBackward".to_string()),
            })
        }

        pub fn transpose(&self) -> Result<Tensor, AugustiumError> {
            // Simplified transpose for 2D tensors
            if self.shape.dims.len() != 2 {
                return Err(AugustiumError::Runtime(
                    "Transpose currently only supports 2D tensors".to_string()
                ));
            }
            
            let result_data = self.data.t().to_owned();
            Ok(Tensor {
                data: result_data,
                shape: TensorShape::new(vec![self.shape.dims[1], self.shape.dims[0]]),
                dtype: self.dtype,
                requires_grad: self.requires_grad,
                grad: None,
                grad_fn: Some("TransposeBackward".to_string()),
            })
        }

        pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Tensor, AugustiumError> {
            let total_elements: usize = new_shape.iter().product();
            if total_elements != self.numel() {
                return Err(AugustiumError::Runtime(
                    format!("Cannot reshape tensor: total elements mismatch ({} vs {})", 
                           total_elements, self.numel())
                ));
            }
            
            let result_data = self.data.clone().into_shape(IxDyn(&new_shape))
                .map_err(|e| AugustiumError::Runtime(format!("Failed to reshape tensor: {}", e)))?;
            
            Ok(Tensor {
                data: result_data,
                shape: TensorShape::new(new_shape),
                dtype: self.dtype,
                requires_grad: self.requires_grad,
                grad: None,
                grad_fn: Some("ReshapeBackward".to_string()),
            })
        }

        pub fn permute(&self, dims: Vec<usize>) -> Result<Tensor, AugustiumError> {
            // Simplified permute implementation - just return a copy for now
            // A full implementation would require complex axis permutation
            Ok(Tensor {
                data: self.data.clone(),
                shape: self.shape.clone(),
                dtype: self.dtype,
                requires_grad: self.requires_grad,
                grad: None,
                grad_fn: Some("PermuteBackward".to_string()),
            })
        }

        pub fn div_scalar(&self, scalar: f32) -> Result<Tensor, AugustiumError> {
            let result_data = &self.data / scalar;
            Ok(Tensor {
                data: result_data,
                shape: self.shape.clone(),
                dtype: self.dtype,
                requires_grad: self.requires_grad,
                grad: None,
                grad_fn: Some("DivScalarBackward".to_string()),
            })
        }

        pub fn grad(&self) -> Option<&Tensor> {
            self.grad.as_ref().map(|g| g.as_ref())
        }

        pub fn zero_grad(&mut self) {
            self.grad = None;
        }

        pub fn to_vec(&self) -> Vec<f32> {
            self.data.iter().cloned().collect()
        }
    }

    impl TensorShape {
        pub fn new(dims: Vec<usize>) -> Self {
            TensorShape { dims }
        }

        pub fn total_elements(&self) -> usize {
            self.dims.iter().product()
        }

        pub fn is_broadcastable(&self, other: &TensorShape) -> bool {
            // Simplified broadcasting check
            self.dims == other.dims
        }
    }

    impl fmt::Display for Tensor {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "Tensor(shape={:?}, dtype={:?})", self.shape.dims, self.dtype)
        }
    }
}

#[cfg(feature = "ml-basic")]
pub use tensor_impl::*;

// Provide stub types when ml-basic feature is not enabled
#[cfg(not(feature = "ml-basic"))]
pub mod tensor_stubs {
    use crate::error::AugustiumError;
    
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum DataType {
        Float32,
    }
    
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct TensorShape {
        pub dims: Vec<usize>,
    }
    
    #[derive(Debug, Clone)]
    pub struct Tensor;
    
    impl Tensor {
        pub fn new(_shape: Vec<usize>, _dtype: DataType) -> Result<Self, AugustiumError> {
            Err(AugustiumError::Runtime("ML basic features not enabled".to_string()))
        }
        
        pub fn zeros(_shape: Vec<usize>) -> Result<Self, AugustiumError> {
            Err(AugustiumError::Runtime("ML basic features not enabled".to_string()))
        }
    }
}

#[cfg(not(feature = "ml-basic"))]
pub use tensor_stubs::*;