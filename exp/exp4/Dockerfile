FROM aider-benchmark

# The upstream image runs as root. FlashTrace runs generated code as the host
# uid, so expose only the two official harness scripts and the Rust toolchain.
RUN chmod 0755 \
      /aider/benchmark \
      /aider/benchmark/cpp-test.sh \
      /aider/benchmark/npm-test.sh \
      /root \
    && chmod -R a+rX /root/.cargo /root/.rustup

ENV RUSTUP_HOME=/root/.rustup
