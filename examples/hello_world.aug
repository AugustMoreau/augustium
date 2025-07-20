// Hello World contract in Augustium
// This is a simple example demonstrating basic Augustium syntax

contract HelloWorld {
    // State variables
    let balance: u256;
    let owner: address;
    
    // Simple function to demonstrate basic arithmetic
    pub fn add_numbers(a: u32, b: u32) -> u32 {
        return a + b;
    }
    
    // Function using blockchain primitives
    pub fn get_sender() -> address {
        return msg.sender;
    }
    
    // Function using block information
    pub fn get_block_number() -> u256 {
        return block.number;
    }
    
    // Function with safety check
    pub fn safe_divide(a: u32, b: u32) -> u32 {
        require(b > 0);
        return a / b;
    }
    
    // Simple greeting function
    pub fn greet() -> u32 {
        return 42;
    }
}