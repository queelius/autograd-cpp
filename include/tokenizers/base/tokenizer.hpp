#pragma once

#include <vector>
#include <string>
#include <unordered_map>

namespace tokenizers {

// Base tokenizer interface
class Tokenizer {
public:
    virtual ~Tokenizer() = default;
    
    // Core tokenization methods
    virtual std::vector<int> encode(const std::string& text) = 0;
    virtual std::string decode(const std::vector<int>& tokens) = 0;
    
    // Batch processing
    virtual std::vector<std::vector<int>> encode_batch(const std::vector<std::string>& texts) {
        std::vector<std::vector<int>> result;
        for (const auto& text : texts) {
            result.push_back(encode(text));
        }
        return result;
    }
    
    // Vocabulary size
    virtual size_t vocab_size() const = 0;
    
    // Special tokens
    virtual int pad_token_id() const { return 0; }
    virtual int unk_token_id() const { return 1; }
    virtual int bos_token_id() const { return 2; }
    virtual int eos_token_id() const { return 3; }
};

// Simple character-level tokenizer for demonstration
class CharTokenizer : public Tokenizer {
private:
    std::unordered_map<char, int> char_to_id_;
    std::unordered_map<int, char> id_to_char_;
    
public:
    CharTokenizer() {
        // Build basic ASCII vocabulary
        for (int i = 0; i < 128; ++i) {
            char c = static_cast<char>(i);
            char_to_id_[c] = i;
            id_to_char_[i] = c;
        }
    }
    
    std::vector<int> encode(const std::string& text) override {
        std::vector<int> tokens;
        for (char c : text) {
            auto it = char_to_id_.find(c);
            if (it != char_to_id_.end()) {
                tokens.push_back(it->second);
            } else {
                tokens.push_back(unk_token_id());
            }
        }
        return tokens;
    }
    
    std::string decode(const std::vector<int>& tokens) override {
        std::string text;
        for (int token : tokens) {
            auto it = id_to_char_.find(token);
            if (it != id_to_char_.end()) {
                text += it->second;
            }
        }
        return text;
    }
    
    size_t vocab_size() const override {
        return char_to_id_.size();
    }
};

} // namespace tokenizers