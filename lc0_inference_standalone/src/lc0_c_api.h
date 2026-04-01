#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

typedef void* Lc0Handle;
typedef void* Lc0BatchHandle;

// Initialize lc0 backend. Returns NULL on failure.
Lc0Handle lc0_init(const char* weights_path, const char* backend_name, int chess960);

// Destroy handle and free resources.
void lc0_destroy(Lc0Handle handle);

// Start a new batch computation.
Lc0BatchHandle lc0_new_batch(Lc0Handle handle);

// Add a FEN position to the batch. Returns the sample index (0-based).
// Legal moves and their canonical indices are cached at this point.
int lc0_batch_add_fen(Lc0BatchHandle batch, const char* fen);

// Run inference on the batch (blocking).
void lc0_batch_compute(Lc0BatchHandle batch);

// Get Q-value for a position (-1 to 1, from STM perspective).
float lc0_batch_get_q(Lc0BatchHandle batch, int sample_idx);

// Get draw probability for a position.
float lc0_batch_get_d(Lc0BatchHandle batch, int sample_idx);

// Get number of legal moves for a position.
int lc0_batch_get_num_moves(Lc0BatchHandle batch, int sample_idx);

// Get all legal move canonical indices and logits in one call.
// out_indices and out_logits must have space for at least lc0_batch_get_num_moves entries.
void lc0_batch_get_moves(Lc0BatchHandle batch, int sample_idx, int* out_indices, float* out_logits);

// Free a batch.
void lc0_free_batch(Lc0BatchHandle batch);

#ifdef __cplusplus
}
#endif
