use std::io::{self, Write};

fn add(a: i128, b: i128) -> i128 { a + b }
fn subtract(a: i128, b: i128) -> i128 { a - b }
fn multiply(a: i128, b: i128) -> i128 { a * b }
fn divide(a: i128, b: i128) -> Option<i128> { if b == 0 { None } else { Some(a / b) } }
fn modulo(a: i128, b: i128) -> Option<i128> { if b == 0 { None } else { Some(a % b) } }
fn power(a: i128, b: u32) -> Option<i128> { if b > 256 { None } else { Some(a.pow(b)) } }
fn abs(a: i128) -> i128 { a.abs() }
fn max(a: i128, b: i128) -> i128 { a.max(b) }
fn min(a: i128, b: i128) -> i128 { a.min(b) }
fn factorial(n: u32) -> Option<i128> {
    if n > 34 { return None; } // 35! > i128::MAX
    let mut res: i128 = 1;
    for i in 1..=n as i128 { res *= i; }
    Some(res)
}

fn main() {
    println!("Simple Calculator CLI (mirrors calculator.aug logic)\nType 'help' for list of commands, 'exit' to quit.");

    loop {
        print!("> ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() { break; }
        let tokens: Vec<&str> = input.trim().split_whitespace().collect();
        if tokens.is_empty() { continue; }

        match tokens[0] {
            "exit" | "quit" => break,
            "help" => {
                println!("Available commands: add sub mul div mod pow abs max min fact help exit");
                println!("Examples: add 3 5  |  pow 2 10  |  fact 6");
                continue;
            }
            "add" if tokens.len()==3 => {
                op2(&tokens, add);
            }
            "sub" | "subtract" if tokens.len()==3 => {
                op2(&tokens, subtract);
            }
            "mul" | "multiply" if tokens.len()==3 => {
                op2(&tokens, multiply);
            }
            "div" | "divide" if tokens.len()==3 => {
                match parse2(&tokens) {
                    Some((a,b)) => match divide(a,b) { Some(r)=>println!("{}",r), None=>println!("Error: divide by zero") },
                    None=>{},
                }
            }
            "mod" | "modulo" if tokens.len()==3 => {
                match parse2(&tokens) {
                    Some((a,b)) => match modulo(a,b) { Some(r)=>println!("{}",r), None=>println!("Error: divide by zero") },
                    None=>{},
                }
            }
            "pow" | "power" if tokens.len()==3 => {
                match parse2(&tokens) {
                    Some((a,b)) => match power(a, b as u32) { Some(r)=>println!("{}",r), None=>println!("Error: exponent too large") },
                    None=>{},
                }
            }
            "abs" if tokens.len()==2 => {
                match tokens[1].parse::<i128>() { Ok(a)=>println!("{}", abs(a)), Err(_)=>println!("Invalid number") }
            }
            "max" if tokens.len()==3 => { op2(&tokens, max); }
            "min" if tokens.len()==3 => { op2(&tokens, min); }
            "fact" | "factorial" if tokens.len()==2 => {
                match tokens[1].parse::<u32>() { Ok(n)=>match factorial(n){Some(r)=>println!("{}",r),None=>println!("Error: n too large")},Err(_)=>println!("Invalid number") }
            }
            _ => println!("Unknown/invalid command. Type 'help'."),
        }
    }
}

fn parse2(tokens: &[&str]) -> Option<(i128,i128)> {
    match (tokens[1].parse::<i128>(), tokens[2].parse::<i128>()) {
        (Ok(a), Ok(b)) => Some((a,b)),
        _ => { println!("Invalid numbers"); None }
    }
}

fn op2<F>(tokens: &[&str], f: F) where F: Fn(i128,i128)->i128 {
    if let Some((a,b)) = parse2(tokens) {
        println!("{}", f(a,b));
    }
}
