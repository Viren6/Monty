#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

typedef void* Lc0Handle;
typedef void* Lc0BatchHandle;

// Initialize lc0 backend. Returns NULL on failure.
// batch_size: fixed batch size for TRT optimization. 0 or negative = variable.
Lc0Handle lc0_init(const char* weights_path, const char* backend_name, int chess960, int batch_size);

// Destroy handle and free resources.
void lc0_destroy(Lc0Handle handle);

// Start a new batch computation.
Lc0BatchHandle lc0_new_batch(Lc0Handle handle);

// Add a FEN position to the batch. Returns the sample index (0-based).
// Legal moves and their canonical indices are cached at this point.
int lc0_batch_add_fen(Lc0BatchHandle batch, const char* fen);

// Run inference on the batch (blocking).
void lc0_batch_compute(Lc0BatchHandle batch);

// Result for a single position (used by bulk extraction).
typedef struct {
    float value;
    float draw;
    int num_moves;
} Lc0SampleHeader;

// Extract all results for the entire batch in one call.
// headers: array of batch_count Lc0SampleHeader structs (written by this function)
// out_indices: flat array for all move indices across all samples
// out_logits: flat array for all move logits across all samples
// Returns total number of moves written across all samples.
// Caller must pre-allocate: sum of all num_moves (or use max_moves_per_position * batch_count).
int lc0_batch_extract_all(Lc0BatchHandle batch, int batch_count,
                          Lc0SampleHeader* headers,
                          int* out_indices, float* out_logits);

// Free a batch.
void lc0_free_batch(Lc0BatchHandle batch);

#ifdef __cplusplus
}
#endif
