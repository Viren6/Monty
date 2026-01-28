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
    float draw = computation.GetDVal(sample_idx);
    
    std::cout << "FEN: " << fen << "\n";
    std::cout << "Value: " << value << "\n";
    std::cout << "Draw: " << draw << "\n";
    
    // Generate Legal Moves
    // Re-parse Position to avoid storing/copying Position objects which might be risky or large.
    Position pos = Position::FromFen(fen);
    const ChessBoard& board = pos.GetBoard();
    MoveList moves = board.GenerateLegalMoves();
    
    // Collect logits for legal moves only
    std::vector<std::pair<int, float>> legal_outputs;
    legal_outputs.reserve(moves.size());
    
    for (const auto& move : moves) {
        // The NN output is rotated by 'transform'.
        // We need to fetch the logit from the NN index corresponding to the transform.
        int nn_idx = MoveToNNIndex(move, transform);
        
        // However, the consumer (datagen) expects indices in the CANONICAL (transform=0) mapping.
        // So we must pair the logit with the canonical index.
        int canonical_idx = MoveToNNIndex(move, 0);

        if (nn_idx >= 0 && nn_idx < 1858 && canonical_idx >= 0) {
             float logit = computation.GetPVal(sample_idx, nn_idx);
             legal_outputs.push_back({canonical_idx, logit});
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
    std::cout.flush(); // Ensure flush for pipes
}

// Helper to separate string by whitespace
std::vector<std::string> Split(const std::string& str) {
    std::istringstream iss(str);
    std::vector<std::string> tokens;
    std::string token;
    while (iss >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

Move ParseMove(const Position& pos, const std::string& move_str, bool chess960) {
    const auto& board = pos.GetBoard();
    MoveList moves = board.GenerateLegalMoves();
    for (const auto& m : moves) {
        Move m_real = m;
        if (pos.IsBlackToMove()) {
            m_real.Flip();
        }
        if (m_real.ToString(chess960) == move_str) return m;
    }
    
    // DEBUG FAILURE
    std::cerr << "FAILED to parse move: " << move_str << "\n";
    std::cerr << "STM: " << (pos.IsBlackToMove() ? "Black" : "White") << "\n";
    std::cerr << "Legal Moves (Internal -> Real): ";
    int count = 0;
    for (const auto& m : moves) {
        Move m_real = m;
        if (pos.IsBlackToMove()) m_real.Flip();
        if (count++ < 10) std::cerr << m_real.ToString(chess960) << " ";
    }
    std::cerr << "... (" << moves.size() << " total)\n";
    
    return Move(); // null
}

int main(int argc, char* argv[]) {
    try {
        if (argc < 2) {
            std::cerr << "Usage: " << argv[0] << " <network_path> [batch_size]\n";
            return 1;
        }

        std::string network_path = argv[1];
        int batch_size = 4;
        bool chess960 = false;

        if (argc >= 3) {
            batch_size = std::stoi(argv[2]);
        }
        
        std::string backend_name;

        // Simple flag parsing
        for (int i = 1; i < argc; ++i) {
             std::string arg = argv[i];
             if (arg == "--chess960") {
                 chess960 = true;
             }
             if (arg == "--backend" && i + 1 < argc) {
                 backend_name = argv[i + 1];
                 i++; // Skip next arg
             }
        }

        InitializeMagicBitboards();
        
        // Load weights
        std::cerr << "Loading network: " << network_path << "\n";
        auto weights = LoadWeightsFromFile(network_path);
        
        // Setup options
        OptionsDict options;
        if (chess960) {
            options.Set<bool>("chess960", true);
            std::cerr << "Enabled Chess960 Mode.\n";
        }

        // Auto-select backend if not specified
        if (backend_name.empty()) {
            auto backends = NetworkFactory::Get()->GetBackendsList();
            if (!backends.empty()) {
                backend_name = backends[0];
                std::cerr << "Auto-selected backend: " << backend_name << "\n";
            } else {
                std::cerr << "No backends found! Ensure you have compiled with backend support.\n";
                return 1;
            }
        } else {
            std::cerr << "Using requested backend: " << backend_name << "\n";
        }

        // Create network
        auto network = NetworkFactory::Get()->Create(backend_name, weights, options);
        
        std::cerr << "Network created. Batch size: " << batch_size << "\n";

        // Interactive loop
        std::vector<std::string> batch_lines;
        batch_lines.reserve(batch_size);
        
        std::string line;
        while (true) {
            batch_lines.clear();
            for (int i = 0; i < batch_size; ++i) {
                 if (std::getline(std::cin, line)) {
                     // Trim is critical for Windows pipes and robustness
                     line = Trim(line);
                     if (!line.empty()) {
                        batch_lines.push_back(line);
                     } else {
                        i--; // retry
                     }
                 } else {
                     if (batch_lines.empty()) return 0;
                     break; 
                 }
            }
            
            if (batch_lines.empty()) break;

            // Process batch
            auto computation = network->NewComputation();
            int current_batch = 0;
            
            // Store transforms for output phase
            std::vector<int> transforms;
            transforms.reserve(batch_lines.size());
            std::vector<std::string> fens; // To store the FINAL fen for checking
            fens.reserve(batch_lines.size());
            
            for (const auto& input_line : batch_lines) {
                // Parse: FEN (6 tokens) + Moves
                auto tokens = Split(input_line);
                if (tokens.size() < 6) {
                    std::cerr << "Error: Invalid input line (too short): " << input_line << "\n";
                    // Hack: push dummy? Or just fail?
                    // Let's just create a startpos to keep alignment
                    tokens = Split("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
                }
                
                std::string start_fen = tokens[0] + " " + tokens[1] + " " + tokens[2] + " " + tokens[3] + " " + tokens[4] + " " + tokens[5];
                
                Position pos = Position::FromFen(start_fen);
                PositionHistory history;
                history.Reset(pos);
                
                for (size_t k = 6; k < tokens.size(); ++k) {
                    Move m = ParseMove(pos, tokens[k], chess960);
                     // If invalid move, we stop processing history to avoid crash, but keep position so far?
                     if (!m.is_null()) {
                         pos = Position(pos, m);
                         history.Append(m);
                     } else {
                         std::cerr << "Warning: Invalid move " << tokens[k] << " for FEN " << start_fen << "\n";
                     }
                }
                
                fens.push_back(PositionToFen(pos));

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
                // Return the FINAL fen, consistent with old behavior validation
                PrintOutput(*computation, k, fens[k], transforms[k]);
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

