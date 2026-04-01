#include "lc0_c_api.h"

#include <string>
#include <vector>
#include <memory>
#include <mutex>

#include "neural/network.h"
#include "neural/factory.h"
#include "neural/encoder.h"
#include "neural/loader.h"
#include "chess/position.h"
#include "chess/board.h"
#include "utils/optionsdict.h"
#include "utils/protomessage.h"

using namespace lczero;

static std::once_flag init_flag;

struct Lc0Context {
    std::unique_ptr<Network> network;
    bool chess960;
};

// Per-position move info cached at add_fen time
struct MoveInfo {
    int nn_idx;        // Index in the NN output (transformed space)
    int canonical_idx; // Index in canonical space (transform=0)
};

struct Lc0Batch {
    Lc0Context* ctx;
    std::unique_ptr<NetworkComputation> computation;
    std::vector<std::vector<MoveInfo>> move_infos; // per-sample
    int count;
};

extern "C" {

Lc0Handle lc0_init(const char* weights_path, const char* backend_name, int chess960) {
    try {
        std::call_once(init_flag, []() {
            InitializeMagicBitboards();
        });

        auto weights = LoadWeightsFromFile(weights_path);

        OptionsDict options;
        if (chess960) {
            options.Set<bool>("chess960", true);
        }

        std::string backend(backend_name);
        if (backend.empty()) {
            auto backends = NetworkFactory::Get()->GetBackendsList();
            if (backends.empty()) return nullptr;
            backend = backends[0];
        }

        auto network = NetworkFactory::Get()->Create(backend, weights, options);
        if (!network) return nullptr;

        auto* ctx = new Lc0Context{std::move(network), chess960 != 0};
        return static_cast<Lc0Handle>(ctx);
    } catch (...) {
        return nullptr;
    }
}

void lc0_destroy(Lc0Handle handle) {
    delete static_cast<Lc0Context*>(handle);
}

Lc0BatchHandle lc0_new_batch(Lc0Handle handle) {
    auto* ctx = static_cast<Lc0Context*>(handle);
    auto* batch = new Lc0Batch{
        ctx,
        ctx->network->NewComputation(),
        {},
        0
    };
    return static_cast<Lc0BatchHandle>(batch);
}

int lc0_batch_add_fen(Lc0BatchHandle batch_handle, const char* fen_str) {
    auto* batch = static_cast<Lc0Batch*>(batch_handle);

    try {
        Position pos = Position::FromFen(fen_str);
        PositionHistory history;
        history.Reset(pos);

        int transform = 0;
        auto input_format = batch->ctx->network->GetCapabilities().input_format;

        InputPlanes planes = EncodePositionForNN(
            input_format,
            history,
            8,
            FillEmptyHistory::FEN_ONLY,
            &transform
        );

        batch->computation->AddInput(std::move(planes));

        // Cache legal moves and their index mappings now
        const ChessBoard& board = pos.GetBoard();
        MoveList moves = board.GenerateLegalMoves();

        std::vector<MoveInfo> infos;
        infos.reserve(moves.size());
        for (const auto& move : moves) {
            int nn_idx = MoveToNNIndex(move, transform);
            int canonical_idx = MoveToNNIndex(move, 0);
            if (nn_idx >= 0 && nn_idx < 1858 && canonical_idx >= 0 && canonical_idx < 1858) {
                infos.push_back({nn_idx, canonical_idx});
            }
        }
        batch->move_infos.push_back(std::move(infos));

        return batch->count++;
    } catch (...) {
        return -1;
    }
}

void lc0_batch_compute(Lc0BatchHandle batch_handle) {
    auto* batch = static_cast<Lc0Batch*>(batch_handle);
    batch->computation->ComputeBlocking();
}

int lc0_batch_extract_all(Lc0BatchHandle batch_handle, int batch_count,
                          Lc0SampleHeader* headers,
                          int* out_indices, float* out_logits) {
    auto* batch = static_cast<Lc0Batch*>(batch_handle);
    int total_moves = 0;

    for (int s = 0; s < batch_count; ++s) {
        headers[s].value = batch->computation->GetQVal(s);
        headers[s].draw = batch->computation->GetDVal(s);

        const auto& infos = batch->move_infos[s];
        int n = static_cast<int>(infos.size());
        headers[s].num_moves = n;

        for (int m = 0; m < n; ++m) {
            out_indices[total_moves + m] = infos[m].canonical_idx;
            out_logits[total_moves + m] = batch->computation->GetPVal(s, infos[m].nn_idx);
        }

        total_moves += n;
    }

    return total_moves;
}

void lc0_free_batch(Lc0BatchHandle batch_handle) {
    delete static_cast<Lc0Batch*>(batch_handle);
}

} // extern "C"
