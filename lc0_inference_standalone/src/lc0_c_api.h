#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

typedef void* Lc0Handle;
typedef void* Lc0BatchHandle;

// Results for a single position
typedef struct {
    float value;     // Q-value (-1 to 1, from STM perspective)
    float draw;      // Draw probability
    int num_moves;   // Number of legal moves with policy
    int* move_indices;   // Array of lc0 policy indices (canonical, transform=0)
    float* move_logits;  // Array of corresponding logits
} Lc0Result;

// Initialize lc0 backend. Returns NULL on failure.
// Must be called once before any other calls. Thread-safe for multiple handles.
Lc0Handle lc0_init(const char* weights_path, const char* backend_name, int chess960);

// Destroy handle and free resources.
void lc0_destroy(Lc0Handle handle);

// Start a new batch computation.
Lc0BatchHandle lc0_new_batch(Lc0Handle handle);

// Add a FEN position to the batch. Returns the sample index (0-based).
// `fen` is the position FEN string.
int lc0_batch_add_fen(Lc0BatchHandle batch, const char* fen);

// Run inference on the batch (blocking).
void lc0_batch_compute(Lc0BatchHandle batch);

// Get the number of positions in the batch.
int lc0_batch_size(Lc0BatchHandle batch);

// Get result for a position. Caller must free the result with lc0_free_result.
Lc0Result lc0_batch_get_result(Lc0BatchHandle batch, int sample_idx);

// Free a result's internal arrays.
void lc0_free_result(Lc0Result* result);

// Free a batch.
void lc0_free_batch(Lc0BatchHandle batch);

#ifdef __cplusplus
}
#endif
