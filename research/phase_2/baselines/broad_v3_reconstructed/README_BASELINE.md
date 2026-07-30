# Broad GraphVAE v3 — reconstructed baseline

This snapshot reconstructs the broad GraphVAE v3 trainer before Task 6B proxy
fine-tuning additions. It was reconstructed from the user-provided edit history
on 2026-07-11; it is not an original Git commit.

The matching saved model is `../../models/vae_model.pt` (epoch 22, objective
version 3). Fine-tuning writes to a separate checkpoint and must not overwrite
that file.
