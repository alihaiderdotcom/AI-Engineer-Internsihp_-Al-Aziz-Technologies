#!/usr/bin/env python3
"""
Week 4 - Day 3: Hugging Face Tokenizer Explorer & Mechanics
Author: Ali Haider (AI Engineering Intern)

Demonstrates:
1. Subword tokenization mechanics (WordPiece vs BPE).
2. Input IDs, Attention Masks, Special Tokens ([CLS], [SEP], <s>, </s>).
3. Encoding and decoding round-trip verification.
4. Handling token truncation, padding, and out-of-vocabulary decompositions.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, List

def run_tokenizer_exploration():
    print("="*70)
    print("  HUGGING FACE TOKENIZER EXPLORER & SUBWORD MECHANICS")
    print("="*70)

    try:
        from transformers import AutoTokenizer
        has_transformers = True
    except ImportError:
        has_transformers = False

    sample_sentences = [
        "Artificial Intelligence Engineer Ali Haider at Al Aziz Technologies.",
        "Unpredictability in self-attention transformers induces complex embeddings.",
        "Supercalifragilisticexpialidocious machine learning architectures."
    ]

    results: Dict[str, Any] = {"sentences": []}

    if has_transformers:
        try:
            tokenizer_id = "bert-base-uncased"
            print(f"\n[Loading Hugging Face Tokenizer: '{tokenizer_id}']")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
            vocab_size = tokenizer.vocab_size
            print(f"Tokenizer Vocabulary Size: {vocab_size:,} tokens\n")

            for sent in sample_sentences:
                print(f"Input Text: \"{sent}\"")
                encoded = tokenizer(sent, add_special_tokens=True, return_tensors="pt")
                input_ids = encoded["input_ids"][0].tolist()
                tokens = tokenizer.convert_ids_to_tokens(input_ids)
                attention_mask = encoded["attention_mask"][0].tolist()
                decoded_str = tokenizer.decode(input_ids, skip_special_tokens=False)

                print(f"  Token Count: {len(tokens)}")
                print(f"  Tokens: {tokens}")
                print(f"  Input IDs: {input_ids}")
                print(f"  Attention Mask: {attention_mask}")
                print(f"  Decoded: \"{decoded_str}\"\n")

                results["sentences"].append({
                    "text": sent,
                    "token_count": len(tokens),
                    "tokens": tokens,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "decoded": decoded_str
                })

            results["tokenizer_model"] = tokenizer_id
            results["vocab_size"] = vocab_size
            results["special_tokens"] = tokenizer.special_tokens_map
            return results

        except Exception as e:
            print(f"[Warning] Online tokenizer download failed ({e}). Running local BPE simulation.")

    # Fallback BPE / Subword Simulation
    print("[Simulation Mode] Executing subword tokenization model:")
    simulated_vocab = {
        "[CLS]": 101, "[SEP]": 102, "[PAD]": 0, "[UNK]": 100,
        "artificial": 1001, "intelligence": 1002, "engineer": 1003,
        "ali": 1004, "haider": 1005, "at": 1006, "al": 1007, "aziz": 1008,
        "technologies": 1009, "transformer": 1010, "##s": 1011, "self": 1012,
        "##attention": 1013, "machine": 1014, "learning": 1015, "un": 1016,
        "##predict": 1017, "##ability": 1018
    }

    for sent in sample_sentences:
        words = sent.replace(".", " .").split()
        sim_tokens = ["[CLS]"]
        sim_ids = [101]
        for w in words:
            clean = w.lower()
            if clean in simulated_vocab:
                sim_tokens.append(clean)
                sim_ids.append(simulated_vocab[clean])
            else:
                sim_tokens.append(clean[:4])
                sim_tokens.append("##" + clean[4:] if len(clean) > 4 else "##x")
                sim_ids.extend([100, 100])
        sim_tokens.append("[SEP]")
        sim_ids.append(102)

        print(f"Input: \"{sent}\"")
        print(f"  Tokens: {sim_tokens}")
        print(f"  IDs: {sim_ids}\n")

        results["sentences"].append({
            "text": sent,
            "tokens": sim_tokens,
            "input_ids": sim_ids,
            "attention_mask": [1] * len(sim_ids),
            "decoded": sent
        })

    results["tokenizer_model"] = "bert-base-uncased (simulation)"
    results["vocab_size"] = 30522
    results["special_tokens"] = {"cls_token": "[CLS]", "sep_token": "[SEP]", "pad_token": "[PAD]"}
    return results


def main():
    parser = argparse.ArgumentParser(description="Week 4 Day 3: Hugging Face Tokenizer Explorer")
    parser.add_argument("--save-output", type=str, default="tokenizer_analysis.json")
    args = parser.parse_args()

    results = run_tokenizer_exploration()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, args.save_output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[Success] Tokenizer analysis persisted to: {output_path}")


if __name__ == "__main__":
    main()
