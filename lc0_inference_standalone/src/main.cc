#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <algorithm>

#include "neural/network.h"
#include "neural/factory.h"
#include "neural/encoder.h"
#include "neural/loader.h"
#include "chess/position.h"
#include "chess/board.h"
#include "utils/optionsdict.h"
#include "utils/protomessage.h"

using namespace lczero;

#include <cmath>

// Helper to trim whitespace
std::string Trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (std::string::npos == first) {
        return str;
    }
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}

// Output only legal moves to optimize bandwidth and parsing speed.
void PrintOutput(NetworkComputation& computation, int sample_idx, const std::string& fen, int transform) {
    float value = computation.GetQVal(sample_idx);
    
    std::cout << "FEN: " << fen << "\n";
    std::cout << "Value: " << value << "\n";
    
    // Generate Legal Moves
    // Re-parse Position to avoid storing/copying Position objects which might be risky or large.
    Position pos = Position::FromFen(fen);
    const ChessBoard& board = pos.GetBoard();
    MoveList moves = board.GenerateLegalMoves();
    
    // Collect logits for legal moves only
    std::vector<std::pair<int, float>> legal_outputs;
    legal_outputs.reserve(moves.size());
    
    for (const auto& move : moves) {
        int idx = MoveToNNIndex(move, transform);
        if (idx >= 0 && idx < 1858) {
             float logit = computation.GetPVal(sample_idx, idx);
             legal_outputs.push_back({idx, logit});
        }
    }
    
    // Sort by index (as requested by user)
    std::sort(legal_outputs.begin(), legal_outputs.end());

    std::cout << "Policy (Logits): ";
    for(const auto& p : legal_outputs) {
         std::cout << p.first << ":" << p.second << " ";
    }
    std::cout << "\n";
    std::cout << "--------------------------------------------------\n";
}

int main(int argc, char* argv[]) {
    try {
        if (argc < 2) {
            std::cerr << "Usage: " << argv[0] << " <network_path> [batch_size]\n";
            return 1;
        }

        std::string network_path = argv[1];
        int batch_size = 4;
        if (argc >= 3) {
            batch_size = std::stoi(argv[2]);
        }

        InitializeMagicBitboards();
        
        // Load weights
        std::cerr << "Loading network: " << network_path << "\n";
        auto weights = LoadWeightsFromFile(network_path);
        
        // Setup options
        OptionsDict options;

        // Auto-select backend
        auto backends = NetworkFactory::Get()->GetBackendsList();
        std::string backend_name;
        if (!backends.empty()) {
            backend_name = backends[0];
            std::cerr << "Auto-selected backend: " << backend_name << "\n";
        } else {
            std::cerr << "No backends found! Ensure you have compiled with backend support.\n";
            return 1;
        }

        // Create network
        auto network = NetworkFactory::Get()->Create(backend_name, weights, options);
        
        std::cerr << "Network created. Batch size: " << batch_size << "\n";

        // Interactive loop
        std::vector<std::string> batch_fens;
        batch_fens.reserve(batch_size);
        
        std::string line;
        while (true) {
            batch_fens.clear();
            for (int i = 0; i < batch_size; ++i) {
                 if (std::getline(std::cin, line)) {
                     // Trim is critical for Windows pipes and robustness
                     line = Trim(line);
                     if (!line.empty()) {
                        batch_fens.push_back(line);
                     } else {
                        i--; // retry
                     }
                 } else {
                     if (batch_fens.empty()) return 0;
                     break; 
                 }
            }
            
            if (batch_fens.empty()) break;

            // Process batch
            auto computation = network->NewComputation();
            int current_batch = 0;
            
            // Store transforms for output phase
            std::vector<int> transforms;
            transforms.reserve(batch_fens.size());
            
            for (const auto& fen : batch_fens) {
                Position pos = Position::FromFen(fen);
                PositionHistory history;
                history.Reset(pos);
                
                int transform = 0; 
                auto input_format = network->GetCapabilities().input_format;
                
                InputPlanes planes = EncodePositionForNN(
                    input_format, 
                    history, 
                    8, // history planes
                    FillEmptyHistory::FEN_ONLY, 
                    &transform
                );
                
                computation->AddInput(std::move(planes));
                transforms.push_back(transform);
                current_batch++;
            }
            
            computation->ComputeBlocking();
            
            for (int k = 0; k < current_batch; ++k) {
                PrintOutput(*computation, k, batch_fens[k], transforms[k]);
            }
            std::cout << "BATCH_DONE\n";
            std::cout.flush();
            
            if (std::cin.eof()) break;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "Unknown error occurred.\n";
        return 1;
    }

    return 0;
}

