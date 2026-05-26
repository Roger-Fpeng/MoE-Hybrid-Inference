"""
Standalone inference for Trinity-Nano-Base with kt-kernel MoE acceleration.
No SGLang required — uses HuggingFace transformers + kt-kernel directly.

Usage:
    python examples/run_afmoe_standalone.py \
        --model-path /path/to/Trinity-Nano-Base \
        --method BF16 \
        --num-gpu-experts 16 \
        --cpuinfer-threads 64 \
        --threadpool-count 2

    # For pure PyTorch fallback (no kt-kernel, for debugging):
    python examples/run_afmoe_standalone.py \
        --model-path /path/to/Trinity-Nano-Base \
        --no-kt-kernel
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

# Add parent directory so we can import the local modeling files
sys.path.insert(0, str(Path(__file__).parent))
from configuration_afmoe import AfmoeConfig
from modeling_afmoe import AfmoeForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(description="Run Trinity-Nano-Base with kt-kernel")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model weights")
    parser.add_argument("--method", type=str, default="BF16",
                        choices=["BF16", "FP8", "AMXINT4", "AMXINT8", "LLAMAFILE"],
                        help="kt-kernel backend method")
    parser.add_argument("--kt-weight-path", type=str, default=None,
                        help="Path to quantized weights (defaults to model-path)")
    parser.add_argument("--num-gpu-experts", type=int, default=16,
                        help="Number of experts to keep on GPU (out of 64)")
    parser.add_argument("--cpuinfer-threads", type=int, default=64,
                        help="Number of CPU inference threads (physical cores)")
    parser.add_argument("--threadpool-count", type=int, default=2,
                        help="Number of NUMA subpools")
    parser.add_argument("--max-deferred", type=int, default=0,
                        help="Deferred experts per token for pipelining")
    parser.add_argument("--no-kt-kernel", action="store_true",
                        help="Disable kt-kernel, use pure PyTorch MoE (for debugging)")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?",
                        help="Input prompt for generation")
    parser.add_argument("--max-new-tokens", type=int, default=128,
                        help="Maximum tokens to generate")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading model from: {args.model_path}")
    print(f"Method: {args.method}, GPU experts: {args.num_gpu_experts}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AfmoeForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()

    # Initialize kt-kernel (if not disabled)
    if not args.no_kt_kernel:
        weight_path = args.kt_weight_path or args.model_path
        print(f"Initializing kt-kernel: method={args.method}, weight_path={weight_path}")
        model.init_kt_kernel(
            method=args.method,
            cpuinfer_threads=args.cpuinfer_threads,
            threadpool_count=args.threadpool_count,
            weight_path=weight_path,
            num_gpu_experts=args.num_gpu_experts,
            max_deferred_experts_per_token=args.max_deferred,
            chunked_prefill_size=512,
        )
        print("kt-kernel initialized successfully!")
    else:
        print("Running in pure PyTorch mode (no kt-kernel)")

    # Tokenize input
    inputs = tokenizer(args.prompt, return_tensors="pt").to(args.device)
    print(f"\nPrompt: {args.prompt}")
    print(f"Input tokens: {inputs['input_ids'].shape[1]}")

    # Generate
    print(f"Generating up to {args.max_new_tokens} tokens...")
    start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )

    elapsed = time.time() - start_time
    generated_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]

    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nGenerated text:\n{generated_text}")
    print(f"\n--- Stats ---")
    print(f"Generated tokens: {generated_tokens}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Throughput: {generated_tokens / elapsed:.1f} tokens/s")


if __name__ == "__main__":
    main()
