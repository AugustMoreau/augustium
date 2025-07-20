# Multi-stage build for Augustium
# Stage 1: Build the Rust application
FROM rust:1.75-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /usr/src/augustium

# Copy Cargo files first for better caching
COPY Cargo.toml Cargo.lock ./

# Create a dummy main.rs to build dependencies
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release && rm -rf src

# Copy the actual source code
COPY . .

# Build the application
RUN cargo build --release

# Stage 2: Create the runtime image
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -u 1000 augustium

# Copy binaries from builder stage
COPY --from=builder /usr/src/augustium/target/release/augustc /usr/local/bin/
COPY --from=builder /usr/src/augustium/target/release/august /usr/local/bin/

# Copy examples and documentation
COPY --from=builder /usr/src/augustium/examples /opt/augustium/examples
COPY --from=builder /usr/src/augustium/docs /opt/augustium/docs
COPY --from=builder /usr/src/augustium/README.md /opt/augustium/

# Set permissions
RUN chmod +x /usr/local/bin/augustc /usr/local/bin/august
RUN chown -R augustium:augustium /opt/augustium

# Switch to non-root user
USER augustium

# Set working directory
WORKDIR /home/augustium

# Set environment variables
ENV PATH="/usr/local/bin:${PATH}"
ENV AUGUSTIUM_HOME="/opt/augustium"

# Add labels
LABEL org.opencontainers.image.title="Augustium"
LABEL org.opencontainers.image.description="The Augustium programming language compiler and virtual machine"
LABEL org.opencontainers.image.url="https://augustium.org"
LABEL org.opencontainers.image.source="https://github.com/AugustMoreau/augustium"
LABEL org.opencontainers.image.version="0.1.0"
LABEL org.opencontainers.image.licenses="MIT"

# Default command
CMD ["august", "--help"]

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD augustc --version || exit 1