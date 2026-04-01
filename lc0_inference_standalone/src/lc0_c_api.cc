#include "lc0_c_api.h"

#include <string>
#include <vector>
#include <memory>
#include <sstream>
#include <cstring>
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

struct Lc0Batch {
    Lc0Context* ctx;
    std::unique_ptr<NetworkComputation> computation;
    std::vector<int> transforms;
    std::vector<std::string> fens;  // final FENs for legal move generation
    int count;
};

static std::vector<std::string> split(const std::string& str) {
    std::istringstream iss(str);
    std::vector<std::string> tokens;
    std::string token;
    while (iss >> token) tokens.push_back(token);
    return tokens;
}

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

        batch->fens.push_back(PositionToFen(pos));

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
        batch->transforms.push_back(transform);

        return batch->count++;
    } catch (...) {
        return -1;
    }
}

void lc0_batch_compute(Lc0BatchHandle batch_handle) {
    auto* batch = static_cast<Lc0Batch*>(batch_handle);
    batch->computation->ComputeBlocking();
}

int lc0_batch_size(Lc0BatchHandle batch_handle) {
    auto* batch = static_cast<Lc0Batch*>(batch_handle);
    return batch->count;
}

Lc0Result lc0_batch_get_result(Lc0BatchHandle batch_handle, int sample_idx) {
    auto* batch = static_cast<Lc0Batch*>(batch_handle);
    Lc0Result result = {};

    try {
        result.value = batch->computation->GetQVal(sample_idx);
        result.draw = batch->computation->GetDVal(sample_idx);

        // Generate legal moves and extract policy logits
        Position pos = Position::FromFen(batch->fens[sample_idx]);
        const ChessBoard& board = pos.GetBoard();
        MoveList moves = board.GenerateLegalMoves();
        int transform = batch->transforms[sample_idx];

        std::vector<int> indices;
        std::vector<float> logits;
        indices.reserve(moves.size());
        logits.reserve(moves.size());

        for (const auto& move : moves) {
            int nn_idx = MoveToNNIndex(move, transform);
            int canonical_idx = MoveToNNIndex(move, 0);

            if (nn_idx >= 0 && nn_idx < 1858 && canonical_idx >= 0) {
                float logit = batch->computation->GetPVal(sample_idx, nn_idx);
                indices.push_back(canonical_idx);
                logits.push_back(logit);
            }
        }

        result.num_moves = static_cast<int>(indices.size());
        if (result.num_moves > 0) {
            result.move_indices = new int[result.num_moves];
            result.move_logits = new float[result.num_moves];
            std::memcpy(result.move_indices, indices.data(), result.num_moves * sizeof(int));
            std::memcpy(result.move_logits, logits.data(), result.num_moves * sizeof(float));
        }
    } catch (...) {
        result.value = 0.0f;
        result.draw = 0.5f;
        result.num_moves = 0;
    }

    return result;
}

void lc0_free_result(Lc0Result* result) {
    if (result) {
        delete[] result->move_indices;
        delete[] result->move_logits;
        result->move_indices = nullptr;
        result->move_logits = nullptr;
        result->num_moves = 0;
    }
}

void lc0_free_batch(Lc0BatchHandle batch_handle) {
    delete static_cast<Lc0Batch*>(batch_handle);
}

} // extern "C"
