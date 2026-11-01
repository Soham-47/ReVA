# ReVA dual-stream temporal grounding

- [x] Implement and review the train-only temporal-target parser/token contract.
- [x] Implement the opt-in InternVideo2 segment stream and Qwen3 fusion path.
- [x] Add save/load and inference support for learned temporal tokens.
- [x] Add an isolated config and usage documentation.
- [x] Run static compilation and focused CPU checks; do not start GPU or downloads.

Runtime follow-up (requires explicit approval and local model dependencies):

- [ ] Run an end-to-end GPU smoke test for Qwen/PEFT cache-aware generation and
  checkpoint restoration; automatic temporal resume remains disabled meanwhile.
