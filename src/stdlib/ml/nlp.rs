//! Natural Language Processing module
//! Includes tokenizers, embeddings, BERT-like models, and text preprocessing

use crate::stdlib::ml::tensor::{Tensor, TensorShape};
use crate::stdlib::ml::deep_learning::{Linear, MultiHeadAttention, LayerNorm, ActivationFunction, Dropout};
use crate::error::AugustiumError;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::File;
use std::io::{BufRead, BufReader};
use tokenizers::{Tokenizer, models::bpe::BPE, pre_tokenizers::byte_level::ByteLevel};
use hf_hub::api::tokio::Api;

/// Token types for special tokens
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SpecialToken {
    Pad,
    Unk,
    Cls,
    Sep,
    Mask,
    Bos,
    Eos,
}

/// Vocabulary for tokenization
#[derive(Debug, Clone)]
pub struct Vocabulary {
    pub token_to_id: HashMap<String, usize>,
    pub id_to_token: HashMap<usize, String>,
    pub special_tokens: HashMap<SpecialToken, usize>,
    pub vocab_size: usize,
}

/// Byte Pair Encoding tokenizer
#[derive(Debug, Clone)]
pub struct BPETokenizer {
    pub vocab: Vocabulary,
    pub merges: Vec<(String, String)>,
    pub max_length: Option<usize>,
    pub padding: bool,
    pub truncation: bool,
}

/// WordPiece tokenizer (BERT-style)
#[derive(Debug, Clone)]
pub struct WordPieceTokenizer {
    pub vocab: Vocabulary,
    pub unk_token: String,
    pub max_input_chars_per_word: usize,
    pub continuing_subword_prefix: String,
}

/// Sentence tokenizer
#[derive(Debug, Clone)]
pub struct SentenceTokenizer {
    pub abbreviations: HashSet<String>,
    pub sentence_endings: HashSet<char>,
}

/// Text preprocessing utilities
#[derive(Debug, Clone)]
pub struct TextPreprocessor {
    pub lowercase: bool,
    pub remove_punctuation: bool,
    pub remove_numbers: bool,
    pub remove_stopwords: bool,
    pub stopwords: HashSet<String>,
    pub min_word_length: usize,
    pub max_word_length: usize,
}

/// N-gram language model
#[derive(Debug, Clone)]
pub struct NGramModel {
    pub n: usize,
    pub vocab: Vocabulary,
    pub counts: HashMap<Vec<usize>, HashMap<usize, usize>>,
    pub smoothing: f32,
}

/// Word embeddings
#[derive(Debug, Clone)]
pub struct WordEmbeddings {
    pub vocab: Vocabulary,
    pub embeddings: Tensor,
    pub embedding_dim: usize,
    pub trainable: bool,
}

/// Positional encoding for transformers
#[derive(Debug, Clone)]
pub struct PositionalEncoding {
    pub max_length: usize,
    pub embedding_dim: usize,
    pub encoding: Tensor,
}

/// BERT-like encoder layer
#[derive(Debug, Clone)]
pub struct BERTEncoderLayer {
    pub attention: MultiHeadAttention,
    pub feed_forward: BERTFeedForward,
    pub norm1: LayerNorm,
    pub norm2: LayerNorm,
    pub dropout: Dropout,
}

/// Feed-forward network for BERT
#[derive(Debug, Clone)]
pub struct BERTFeedForward {
    pub linear1: Linear,
    pub linear2: Linear,
    pub dropout: Dropout,
    pub activation: ActivationFunction,
}

/// BERT model
#[derive(Debug, Clone)]
pub struct BERT {
    pub embeddings: BERTEmbeddings,
    pub encoder_layers: Vec<BERTEncoderLayer>,
    pub pooler: Linear,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
}

/// BERT embeddings (token + position + segment)
#[derive(Debug, Clone)]
pub struct BERTEmbeddings {
    pub token_embeddings: WordEmbeddings,
    pub position_embeddings: PositionalEncoding,
    pub segment_embeddings: Tensor,
    pub layer_norm: LayerNorm,
    pub dropout: Dropout,
}

/// GPT-style decoder layer
#[derive(Debug, Clone)]
pub struct GPTDecoderLayer {
    pub self_attention: MultiHeadAttention,
    pub feed_forward: BERTFeedForward,
    pub norm1: LayerNorm,
    pub norm2: LayerNorm,
}

/// GPT model
#[derive(Debug, Clone)]
pub struct GPT {
    pub embeddings: WordEmbeddings,
    pub position_embeddings: PositionalEncoding,
    pub decoder_layers: Vec<GPTDecoderLayer>,
    pub layer_norm: LayerNorm,
    pub lm_head: Linear,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
}

/// Text classification head
#[derive(Debug, Clone)]
pub struct TextClassificationHead {
    pub dropout: Dropout,
    pub classifier: Linear,
    pub num_classes: usize,
}

/// Named Entity Recognition head
#[derive(Debug, Clone)]
pub struct NERHead {
    pub dropout: Dropout,
    pub classifier: Linear,
    pub num_labels: usize,
}

/// Question Answering head
#[derive(Debug, Clone)]
pub struct QAHead {
    pub qa_outputs: Linear,
}

/// Text similarity metrics
#[derive(Debug, Clone)]
pub struct TextSimilarity;

/// Attention visualization
#[derive(Debug, Clone)]
pub struct AttentionVisualizer {
    pub attention_weights: Vec<Tensor>,
    pub tokens: Vec<String>,
}

/// Vocabulary implementation
impl Vocabulary {
    pub fn new() -> Self {
        let mut vocab = Vocabulary {
            token_to_id: HashMap::new(),
            id_to_token: HashMap::new(),
            special_tokens: HashMap::new(),
            vocab_size: 0,
        };
        
        // Add special tokens
        vocab.add_special_token(SpecialToken::Pad, "[PAD]");
        vocab.add_special_token(SpecialToken::Unk, "[UNK]");
        vocab.add_special_token(SpecialToken::Cls, "[CLS]");
        vocab.add_special_token(SpecialToken::Sep, "[SEP]");
        vocab.add_special_token(SpecialToken::Mask, "[MASK]");
        
        vocab
    }
    
    pub fn add_special_token(&mut self, token_type: SpecialToken, token: &str) {
        let id = self.vocab_size;
        self.token_to_id.insert(token.to_string(), id);
        self.id_to_token.insert(id, token.to_string());
        self.special_tokens.insert(token_type, id);
        self.vocab_size += 1;
    }
    
    pub fn add_token(&mut self, token: &str) -> usize {
        if let Some(&id) = self.token_to_id.get(token) {
            return id;
        }
        
        let id = self.vocab_size;
        self.token_to_id.insert(token.to_string(), id);
        self.id_to_token.insert(id, token.to_string());
        self.vocab_size += 1;
        id
    }
    
    pub fn get_id(&self, token: &str) -> usize {
        self.token_to_id.get(token)
            .copied()
            .unwrap_or_else(|| self.special_tokens[&SpecialToken::Unk])
    }
    
    pub fn get_token(&self, id: usize) -> Option<&String> {
        self.id_to_token.get(&id)
    }
    
    pub fn from_file(file_path: &str) -> Result<Self, AugustiumError> {
        let mut vocab = Vocabulary::new();
        let file = File::open(file_path)
            .map_err(|e| AugustiumError::Runtime(format!("Failed to open vocab file: {}", e)))?;
        
        let reader = BufReader::new(file);
        for line in reader.lines() {
            let line = line.map_err(|e| AugustiumError::Runtime(format!("Failed to read line: {}", e)))?;
            let token = line.trim();
            if !token.is_empty() {
                vocab.add_token(token);
            }
        }
        
        Ok(vocab)
    }
}

/// BPE Tokenizer implementation
impl BPETokenizer {
    pub fn new(vocab: Vocabulary, merges: Vec<(String, String)>) -> Self {
        BPETokenizer {
            vocab,
            merges,
            max_length: None,
            padding: false,
            truncation: false,
        }
    }
    
    pub fn from_pretrained(model_name: &str) -> Result<Self, AugustiumError> {
        // Load from Hugging Face Hub
        let api = Api::new().map_err(|e| AugustiumError::Runtime(format!("Failed to create API: {}", e)))?;
        let repo = api.model(model_name.to_string());
        
        // This is a simplified version - would need proper async handling
        let vocab = Vocabulary::new(); // Placeholder
        let merges = Vec::new(); // Placeholder
        
        Ok(BPETokenizer::new(vocab, merges))
    }
    
    pub fn encode(&self, text: &str) -> Result<Vec<usize>, AugustiumError> {
        let tokens = self.tokenize(text)?;
        Ok(tokens.into_iter().map(|token| self.vocab.get_id(&token)).collect())
    }
    
    pub fn decode(&self, ids: &[usize]) -> Result<String, AugustiumError> {
        let tokens: Result<Vec<_>, _> = ids.iter()
            .map(|&id| self.vocab.get_token(id)
                .ok_or_else(|| AugustiumError::Runtime(format!("Unknown token id: {}", id))))
            .collect();
        
        let tokens = tokens?;
        Ok(tokens.join(" "))
    }
    
    fn tokenize(&self, text: &str) -> Result<Vec<String>, AugustiumError> {
        // Simplified BPE tokenization
        let mut tokens = text.split_whitespace()
            .map(|s| s.chars().map(|c| c.to_string()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        
        // Apply BPE merges
        for (first, second) in &self.merges {
            for token_chars in &mut tokens {
                let mut i = 0;
                while i < token_chars.len() - 1 {
                    if token_chars[i] == *first && token_chars[i + 1] == *second {
                        let merged = format!("{}{}", first, second);
                        token_chars[i] = merged;
                        token_chars.remove(i + 1);
                    } else {
                        i += 1;
                    }
                }
            }
        }
        
        Ok(tokens.into_iter().flatten().collect())
    }
    
    pub fn encode_batch(&self, texts: &[String]) -> Result<Vec<Vec<usize>>, AugustiumError> {
        texts.iter().map(|text| self.encode(text)).collect()
    }
}

/// WordPiece tokenizer implementation
impl WordPieceTokenizer {
    pub fn new(vocab: Vocabulary) -> Self {
        WordPieceTokenizer {
            vocab,
            unk_token: "[UNK]".to_string(),
            max_input_chars_per_word: 100,
            continuing_subword_prefix: "##".to_string(),
        }
    }
    
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        let mut output_tokens = Vec::new();
        
        for word in text.split_whitespace() {
            if word.chars().count() > self.max_input_chars_per_word {
                output_tokens.push(self.unk_token.clone());
                continue;
            }
            
            let mut is_bad = false;
            let mut start = 0;
            let mut sub_tokens = Vec::new();
            
            while start < word.len() {
                let mut end = word.len();
                let mut cur_substr = None;
                
                while start < end {
                    let mut substr = word[start..end].to_string();
                    if start > 0 {
                        substr = format!("{}{}", self.continuing_subword_prefix, substr);
                    }
                    
                    if self.vocab.token_to_id.contains_key(&substr) {
                        cur_substr = Some(substr);
                        break;
                    }
                    end -= 1;
                }
                
                if cur_substr.is_none() {
                    is_bad = true;
                    break;
                }
                
                sub_tokens.push(cur_substr.unwrap());
                start = end;
            }
            
            if is_bad {
                output_tokens.push(self.unk_token.clone());
            } else {
                output_tokens.extend(sub_tokens);
            }
        }
        
        output_tokens
    }
}

/// Text preprocessing implementation
impl TextPreprocessor {
    pub fn new() -> Self {
        let stopwords = [
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
            "to", "was", "will", "with", "the", "this", "but", "they", "have",
            "had", "what", "said", "each", "which", "their", "time", "if"
        ].iter().map(|s| s.to_string()).collect();
        
        TextPreprocessor {
            lowercase: true,
            remove_punctuation: true,
            remove_numbers: false,
            remove_stopwords: false,
            stopwords,
            min_word_length: 1,
            max_word_length: 50,
        }
    }
    
    pub fn process(&self, text: &str) -> String {
        let mut processed = text.to_string();
        
        // Lowercase
        if self.lowercase {
            processed = processed.to_lowercase();
        }
        
        // Remove punctuation
        if self.remove_punctuation {
            processed = processed.chars()
                .filter(|c| c.is_alphanumeric() || c.is_whitespace())
                .collect();
        }
        
        // Remove numbers
        if self.remove_numbers {
            processed = processed.chars()
                .filter(|c| !c.is_numeric())
                .collect();
        }
        
        // Filter words
        let words: Vec<String> = processed.split_whitespace()
            .filter(|word| {
                let len = word.len();
                len >= self.min_word_length && len <= self.max_word_length
            })
            .filter(|word| {
                !self.remove_stopwords || !self.stopwords.contains(*word)
            })
            .map(|s| s.to_string())
            .collect();
        
        words.join(" ")
    }
    
    pub fn process_batch(&self, texts: &[String]) -> Vec<String> {
        texts.iter().map(|text| self.process(text)).collect()
    }
}

/// Word embeddings implementation
impl WordEmbeddings {
    pub fn new(vocab: Vocabulary, embedding_dim: usize) -> Result<Self, AugustiumError> {
        let embeddings = Tensor::randn(vec![vocab.vocab_size, embedding_dim], 0.0, 0.1)?;
        
        Ok(WordEmbeddings {
            vocab,
            embeddings,
            embedding_dim,
            trainable: true,
        })
    }
    
    pub fn from_pretrained(file_path: &str) -> Result<Self, AugustiumError> {
        // Load pre-trained embeddings (e.g., GloVe, Word2Vec)
        let file = File::open(file_path)
            .map_err(|e| AugustiumError::Runtime(format!("Failed to open embeddings file: {}", e)))?;
        
        let reader = BufReader::new(file);
        let mut vocab = Vocabulary::new();
        let mut embedding_vectors = Vec::new();
        let mut embedding_dim = 0;
        
        for (i, line) in reader.lines().enumerate() {
            let line = line.map_err(|e| AugustiumError::Runtime(format!("Failed to read line: {}", e)))?;
            let parts: Vec<&str> = line.trim().split_whitespace().collect();
            
            if parts.is_empty() {
                continue;
            }
            
            let word = parts[0];
            let vector: Result<Vec<f32>, _> = parts[1..].iter()
                .map(|s| s.parse::<f32>())
                .collect();
            
            match vector {
                Ok(vec) => {
                    if i == 0 {
                        embedding_dim = vec.len();
                    } else if vec.len() != embedding_dim {
                        return Err(AugustiumError::Runtime(
                            "Inconsistent embedding dimensions".to_string()
                        ));
                    }
                    
                    vocab.add_token(word);
                    embedding_vectors.extend(vec);
                },
                Err(_) => {
                    return Err(AugustiumError::Runtime(
                        format!("Failed to parse embedding vector for word: {}", word)
                    ));
                }
            }
        }
        
        let embeddings = Tensor::from_data(embedding_vectors, vec![vocab.vocab_size, embedding_dim])?;
        
        Ok(WordEmbeddings {
            vocab,
            embeddings,
            embedding_dim,
            trainable: false,
        })
    }
    
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor, AugustiumError> {
        // Embedding lookup
        let input_shape = input_ids.shape();
        let input_data = input_ids.to_vec();
        let embedding_data = self.embeddings.to_vec();
        
        let mut output_data = Vec::new();
        
        for &id in &input_data {
            let id = id as usize;
            if id >= self.vocab.vocab_size {
                return Err(AugustiumError::Runtime(
                    format!("Token id {} out of vocabulary range", id)
                ));
            }
            
            let start_idx = id * self.embedding_dim;
            let end_idx = start_idx + self.embedding_dim;
            output_data.extend_from_slice(&embedding_data[start_idx..end_idx]);
        }
        
        let mut output_shape = input_shape.dims.clone();
        output_shape.push(self.embedding_dim);
        
        Tensor::from_data(output_data, output_shape)
    }
    
    pub fn get_word_vector(&self, word: &str) -> Option<Tensor> {
        if let Some(&id) = self.vocab.token_to_id.get(word) {
            let embedding_data = self.embeddings.to_vec();
            let start_idx = id * self.embedding_dim;
            let end_idx = start_idx + self.embedding_dim;
            
            Tensor::from_data(
                embedding_data[start_idx..end_idx].to_vec(),
                vec![self.embedding_dim]
            ).ok()
        } else {
            None
        }
    }
}

/// Positional encoding implementation
impl PositionalEncoding {
    pub fn new(max_length: usize, embedding_dim: usize) -> Result<Self, AugustiumError> {
        let mut encoding_data = vec![0.0f32; max_length * embedding_dim];
        
        for pos in 0..max_length {
            for i in 0..embedding_dim {
                let angle = pos as f32 / 10000.0_f32.powf(2.0 * (i / 2) as f32 / embedding_dim as f32);
                
                if i % 2 == 0 {
                    encoding_data[pos * embedding_dim + i] = angle.sin();
                } else {
                    encoding_data[pos * embedding_dim + i] = angle.cos();
                }
            }
        }
        
        let encoding = Tensor::from_data(encoding_data, vec![max_length, embedding_dim])?;
        
        Ok(PositionalEncoding {
            max_length,
            embedding_dim,
            encoding,
        })
    }
    
    pub fn forward(&self, input: &Tensor) -> Result<Tensor, AugustiumError> {
        let input_shape = input.shape();
        let seq_length = input_shape.dims[input_shape.dims.len() - 2];
        
        if seq_length > self.max_length {
            return Err(AugustiumError::Runtime(
                format!("Sequence length {} exceeds maximum length {}", seq_length, self.max_length)
            ));
        }
        
        // Add positional encoding to input
        let pos_encoding = self.encoding.slice_tensor(0, seq_length)?;
        input.add(&pos_encoding)
    }
}

/// BERT implementation
impl BERT {
    pub fn new(vocab_size: usize, hidden_size: usize, num_layers: usize, 
               num_attention_heads: usize, max_position_embeddings: usize) -> Result<Self, AugustiumError> {
        let vocab = Vocabulary::new(); // Would need proper vocab
        let embeddings = BERTEmbeddings::new(vocab, hidden_size, max_position_embeddings)?;
        
        let mut encoder_layers = Vec::new();
        for _ in 0..num_layers {
            encoder_layers.push(BERTEncoderLayer::new(hidden_size, num_attention_heads)?);
        }
        
        let pooler = Linear::new(hidden_size, hidden_size, true)?;
        
        Ok(BERT {
            embeddings,
            encoder_layers,
            pooler,
            vocab_size,
            hidden_size,
            num_layers,
            num_attention_heads,
        })
    }
    
    pub fn forward(&mut self, input_ids: &Tensor, attention_mask: Option<&Tensor>, 
                   token_type_ids: Option<&Tensor>) -> Result<Tensor, AugustiumError> {
        // Embeddings
        let mut hidden_states = self.embeddings.forward(input_ids, token_type_ids)?;
        
        // Encoder layers
        for layer in &mut self.encoder_layers {
            hidden_states = layer.forward(&hidden_states, attention_mask)?;
        }
        
        // Pooling (use [CLS] token)
        let cls_hidden = hidden_states.slice_tensor(0, 1)?; // First token
        self.pooler.forward(&cls_hidden)
    }
    
    pub fn from_pretrained(model_name: &str) -> Result<Self, AugustiumError> {
        // Load pre-trained BERT model
        // This would involve downloading weights from Hugging Face Hub
        BERT::new(30522, 768, 12, 12, 512) // BERT-base configuration
    }
}

/// BERT embeddings implementation
impl BERTEmbeddings {
    pub fn new(vocab: Vocabulary, hidden_size: usize, max_position_embeddings: usize) -> Result<Self, AugustiumError> {
        let token_embeddings = WordEmbeddings::new(vocab, hidden_size)?;
        let position_embeddings = PositionalEncoding::new(max_position_embeddings, hidden_size)?;
        let segment_embeddings = Tensor::randn(vec![2, hidden_size], 0.0, 0.02)?; // 2 segment types
        let layer_norm = LayerNorm::new(vec![hidden_size], 1e-12)?;
        let dropout = Dropout::new(0.1);
        
        Ok(BERTEmbeddings {
            token_embeddings,
            position_embeddings,
            segment_embeddings,
            layer_norm,
            dropout,
        })
    }
    
    pub fn forward(&mut self, input_ids: &Tensor, token_type_ids: Option<&Tensor>) -> Result<Tensor, AugustiumError> {
        let token_embeds = self.token_embeddings.forward(input_ids)?;
        let position_embeds = self.position_embeddings.forward(&token_embeds)?;
        
        let mut embeddings = token_embeds.add(&position_embeds)?;
        
        // Add segment embeddings if provided
        if let Some(segment_ids) = token_type_ids {
            // Simplified segment embedding lookup
            embeddings = embeddings.add(&self.segment_embeddings)?;
        }
        
        let embeddings = self.layer_norm.forward(&embeddings)?;
        self.dropout.forward(&embeddings)
    }
}

/// BERT encoder layer implementation
impl BERTEncoderLayer {
    pub fn new(hidden_size: usize, num_attention_heads: usize) -> Result<Self, AugustiumError> {
        let attention = MultiHeadAttention::new(hidden_size, num_attention_heads, 0.1)?;
        let feed_forward = BERTFeedForward::new(hidden_size, hidden_size * 4)?;
        let norm1 = LayerNorm::new(vec![hidden_size], 1e-12)?;
        let norm2 = LayerNorm::new(vec![hidden_size], 1e-12)?;
        let dropout = Dropout::new(0.1);
        
        Ok(BERTEncoderLayer {
            attention,
            feed_forward,
            norm1,
            norm2,
            dropout,
        })
    }
    
    pub fn forward(&mut self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor, AugustiumError> {
        // Self-attention with residual connection
        let attention_output = self.attention.forward(hidden_states, hidden_states, hidden_states, attention_mask)?;
        let attention_output = self.dropout.forward(&attention_output)?;
        let hidden_states = self.norm1.forward(&hidden_states.add(&attention_output)?)?;
        
        // Feed-forward with residual connection
        let ff_output = self.feed_forward.forward(&hidden_states)?;
        let ff_output = self.dropout.forward(&ff_output)?;
        self.norm2.forward(&hidden_states.add(&ff_output)?)
    }
}

/// BERT feed-forward implementation
impl BERTFeedForward {
    pub fn new(hidden_size: usize, intermediate_size: usize) -> Result<Self, AugustiumError> {
        let linear1 = Linear::new(hidden_size, intermediate_size, true)?;
        let linear2 = Linear::new(intermediate_size, hidden_size, true)?;
        let dropout = Dropout::new(0.1);
        
        Ok(BERTFeedForward {
            linear1,
            linear2,
            dropout,
            activation: ActivationFunction::GELU,
        })
    }
    
    pub fn forward(&mut self, input: &Tensor) -> Result<Tensor, AugustiumError> {
        let x = self.linear1.forward(input)?;
        let x = self.activation.apply(&x)?;
        let x = self.dropout.forward(&x)?;
        self.linear2.forward(&x)
    }
}

/// Text similarity implementation
impl TextSimilarity {
    /// Cosine similarity between two text embeddings
    pub fn cosine_similarity(embedding1: &Tensor, embedding2: &Tensor) -> Result<f32, AugustiumError> {
        let dot_product = embedding1.mul(embedding2)?.sum(None, false)?;
        let norm1 = embedding1.mul(embedding1)?.sum(None, false)?.sqrt()?;
        let norm2 = embedding2.mul(embedding2)?.sum(None, false)?.sqrt()?;
        
        let similarity = dot_product.div(&norm1.mul(&norm2)?)?;
        Ok(similarity.to_vec()[0])
    }
    
    /// Euclidean distance between embeddings
    pub fn euclidean_distance(embedding1: &Tensor, embedding2: &Tensor) -> Result<f32, AugustiumError> {
        let diff = embedding1.sub(embedding2)?;
        let squared_diff = diff.mul(&diff)?;
        let distance = squared_diff.sum(None, false)?.sqrt()?;
        Ok(distance.to_vec()[0])
    }
    
    /// Jaccard similarity for token sets
    pub fn jaccard_similarity(tokens1: &[String], tokens2: &[String]) -> f32 {
        let set1: HashSet<_> = tokens1.iter().collect();
        let set2: HashSet<_> = tokens2.iter().collect();
        
        let intersection = set1.intersection(&set2).count();
        let union = set1.union(&set2).count();
        
        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }
}

/// Text generation utilities
pub struct TextGenerator {
    pub model: GPT,
    pub tokenizer: BPETokenizer,
    pub max_length: usize,
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
}

impl TextGenerator {
    pub fn new(model: GPT, tokenizer: BPETokenizer) -> Self {
        TextGenerator {
            model,
            tokenizer,
            max_length: 100,
            temperature: 1.0,
            top_k: None,
            top_p: None,
        }
    }
    
    pub fn generate(&mut self, prompt: &str) -> Result<String, AugustiumError> {
        let input_ids = self.tokenizer.encode(prompt)?;
        let mut generated_ids = input_ids.clone();
        
        for _ in 0..self.max_length {
            let input_tensor = Tensor::from_data(
                generated_ids.iter().map(|&x| x as f32).collect(),
                vec![1, generated_ids.len()]
            )?;
            
            let logits = self.model.forward(&input_tensor)?;
            let next_token = self.sample_next_token(&logits)?;
            
            generated_ids.push(next_token);
            
            // Check for end token
            if next_token == self.tokenizer.vocab.special_tokens[&SpecialToken::Eos] {
                break;
            }
        }
        
        self.tokenizer.decode(&generated_ids[input_ids.len()..])
    }
    
    fn sample_next_token(&self, logits: &Tensor) -> Result<usize, AugustiumError> {
        // Apply temperature
        let scaled_logits = logits.div_scalar(self.temperature)?;
        
        // Apply top-k filtering
        let mut probs = if let Some(k) = self.top_k {
            self.top_k_filtering(&scaled_logits, k)?
        } else {
            scaled_logits
        };
        
        // Apply top-p filtering
        if let Some(p) = self.top_p {
            probs = self.top_p_filtering(&probs, p)?;
        }
        
        // Apply softmax
        let probs = ActivationFunction::Softmax.apply(&probs)?;
        
        // Sample from distribution
        self.multinomial_sample(&probs)
    }
    
    fn top_k_filtering(&self, logits: &Tensor, k: usize) -> Result<Tensor, AugustiumError> {
        // Simplified top-k filtering
        Ok(logits.clone()) // Placeholder
    }
    
    fn top_p_filtering(&self, logits: &Tensor, p: f32) -> Result<Tensor, AugustiumError> {
        // Simplified top-p filtering
        Ok(logits.clone()) // Placeholder
    }
    
    fn multinomial_sample(&self, probs: &Tensor) -> Result<usize, AugustiumError> {
        let prob_vec = probs.to_vec();
        let random_val = fastrand::f32();
        let mut cumsum = 0.0;
        
        for (i, &prob) in prob_vec.iter().enumerate() {
            cumsum += prob;
            if random_val <= cumsum {
                return Ok(i);
            }
        }
        
        Ok(prob_vec.len() - 1)
    }
}

/// GPT implementation
impl GPT {
    pub fn new(vocab_size: usize, hidden_size: usize, num_layers: usize, 
               max_position_embeddings: usize) -> Result<Self, AugustiumError> {
        let vocab = Vocabulary::new();
        let embeddings = WordEmbeddings::new(vocab, hidden_size)?;
        let position_embeddings = PositionalEncoding::new(max_position_embeddings, hidden_size)?;
        
        let mut decoder_layers = Vec::new();
        for _ in 0..num_layers {
            decoder_layers.push(GPTDecoderLayer::new(hidden_size, 12)?); // 12 attention heads
        }
        
        let layer_norm = LayerNorm::new(vec![hidden_size], 1e-5)?;
        let lm_head = Linear::new(hidden_size, vocab_size, false)?;
        
        Ok(GPT {
            embeddings,
            position_embeddings,
            decoder_layers,
            layer_norm,
            lm_head,
            vocab_size,
            hidden_size,
            num_layers,
        })
    }
    
    pub fn forward(&mut self, input_ids: &Tensor) -> Result<Tensor, AugustiumError> {
        let token_embeds = self.embeddings.forward(input_ids)?;
        let mut hidden_states = self.position_embeddings.forward(&token_embeds)?;
        
        for layer in &mut self.decoder_layers {
            hidden_states = layer.forward(&hidden_states)?;
        }
        
        let hidden_states = self.layer_norm.forward(&hidden_states)?;
        self.lm_head.forward(&hidden_states)
    }
}

/// GPT decoder layer implementation
impl GPTDecoderLayer {
    pub fn new(hidden_size: usize, num_attention_heads: usize) -> Result<Self, AugustiumError> {
        let self_attention = MultiHeadAttention::new(hidden_size, num_attention_heads, 0.1)?;
        let feed_forward = BERTFeedForward::new(hidden_size, hidden_size * 4)?;
        let norm1 = LayerNorm::new(vec![hidden_size], 1e-5)?;
        let norm2 = LayerNorm::new(vec![hidden_size], 1e-5)?;
        
        Ok(GPTDecoderLayer {
            self_attention,
            feed_forward,
            norm1,
            norm2,
        })
    }
    
    pub fn forward(&mut self, hidden_states: &Tensor) -> Result<Tensor, AugustiumError> {
        // Self-attention with residual connection and pre-norm
        let normed = self.norm1.forward(hidden_states)?;
        let attention_output = self.self_attention.forward(&normed, &normed, &normed, None)?;
        let hidden_states = hidden_states.add(&attention_output)?;
        
        // Feed-forward with residual connection and pre-norm
        let normed = self.norm2.forward(&hidden_states)?;
        let ff_output = self.feed_forward.forward(&normed)?;
        hidden_states.add(&ff_output)
    }
}