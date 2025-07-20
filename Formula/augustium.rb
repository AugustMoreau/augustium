class Augustium < Formula
  desc "The Augustium programming language compiler and virtual machine for blockchain smart contracts"
  homepage "https://augustium.org"
  url "https://github.com/AugustMoreau/augustium/archive/v0.1.0.tar.gz"
  sha256 "" # This will be filled automatically by Homebrew when you create the formula
  license "MIT"
  head "https://github.com/AugustMoreau/augustium.git", branch: "main"

  depends_on "rust" => :build

  def install
    system "cargo", "install", *std_cargo_args
    
    # Install additional files
    pkgshare.install "examples"
    pkgshare.install "docs"
    
    # Generate shell completions if available
    if (buildpath/"completions").exist?
      bash_completion.install "completions/august.bash" => "august"
      zsh_completion.install "completions/_august"
      fish_completion.install "completions/august.fish"
    end
  end

  test do
    # Test that the binaries are installed and working
    assert_match "augustc", shell_output("#{bin}/augustc --version")
    assert_match "august", shell_output("#{bin}/august --version")
    
    # Test basic compilation
    (testpath/"test.aug").write <<~EOS
      contract HelloWorld {
          public fn greet() -> string {
              return "Hello, World!";
          }
      }
    EOS
    
    system "#{bin}/augustc", "test.aug"
    assert_predicate testpath/"test.avm", :exist?
  end
end