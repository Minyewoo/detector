FROM --platform=arm64 rust:1.73.0-alpine3.18
RUN apk add --no-cache musl-dev opencv-dev clang-dev build-base
COPY src ./src
COPY Cargo.toml Cargo.lock ./
RUN RUST_BACKTRACE=full RUSTFLAGS="-C target-feature=-crt-static" cargo build -vv --release