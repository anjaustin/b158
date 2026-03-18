#!/usr/bin/env python3
"""
Floyd-Steinberg Chat Module
A standalone Python module for testing Floyd-Steinberg dithered inference with chat.

Usage:
    python fs_chat.py                    # Interactive chat with defaults
    python fs_chat.py --fs 0.3           # Set F-S intensity
    python fs_chat.py --temp 0.5         # Set temperature
    python fs_chat.py --system "You are a poet"  # Custom system prompt
"""

import os
import sys
import subprocess
import argparse
import re
from typing import Optional, Tuple

# Default paths
DEFAULT_MODEL = "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
DEFAULT_BINARY = "build/bin/llama-cli"


class FSChat:
    """Floyd-Steinberg Chat Interface"""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        binary_path: str = DEFAULT_BINARY,
        system_prompt: str = "You are a friend.",
        fs_intensity: float = 0.04376,
        dither_intensity: float = 0.432,
        temperature: float = 0.8645,
        max_tokens: int = 2048,
        threads: int = 8,
        ctx_size: int = 4096,
    ):
        self.model_path = model_path
        self.binary_path = binary_path
        self.system_prompt = system_prompt
        self.fs_intensity = fs_intensity
        self.dither_intensity = dither_intensity
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.threads = threads
        self.ctx_size = ctx_size

        # Validate paths
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        if not os.path.exists(self.binary_path):
            raise FileNotFoundError(f"Binary not found: {self.binary_path}")

    def format_prompt(self, user_message: str, history: list = None) -> str:
        """Format a prompt using the Llama-3 chat template."""
        # BitNet-b1.58-2B-4T uses Llama-3 instruct format
        parts = []

        # System message
        parts.append(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{self.system_prompt}<|eot_id|>")

        # History (if any)
        if history:
            for msg in history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>")

        # Current user message
        parts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{user_message}<|eot_id|>")

        # Assistant header (model will complete)
        parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")

        return "".join(parts)

    def parse_response(self, raw_output: str) -> str:
        """Parse the raw output from llama-cli to extract just the assistant response."""
        # The output includes the prompt echo, we need to find the assistant's actual response

        # Look for content after the last assistant header
        marker = "<|start_header_id|>assistant<|end_header_id|>"
        if marker in raw_output:
            response = raw_output.split(marker)[-1]
        else:
            # Try alternate format (sometimes shows as separate lines)
            if "assistant\n\n" in raw_output:
                parts = raw_output.split("assistant\n\n")
                response = parts[-1] if len(parts) > 1 else raw_output
            else:
                response = raw_output

        # Clean up special tokens
        response = re.sub(r'<\|eot_id\|>.*', '', response, flags=re.DOTALL)
        response = re.sub(r'<\|end_of_text\|>.*', '', response, flags=re.DOTALL)
        response = re.sub(r'\[end of text\].*', '', response, flags=re.DOTALL)

        # Remove performance stats that llama-cli prints
        response = re.sub(r'\nllama_perf.*', '', response, flags=re.DOTALL)
        response = re.sub(r'\nmain:.*', '', response, flags=re.DOTALL)
        response = re.sub(r'\nsampler.*', '', response, flags=re.DOTALL)
        response = re.sub(r'\ngenerate:.*', '', response, flags=re.DOTALL)

        # Remove leading/trailing whitespace and newlines
        response = response.strip()

        # Remove leftover header fragments
        response = re.sub(r'^system\s*\n.*?user\s*\n.*?assistant\s*\n', '', response, flags=re.DOTALL)

        return response.strip()

    def generate(self, user_message: str, history: list = None) -> Tuple[str, str]:
        """
        Generate a response to a user message.

        Returns:
            Tuple of (parsed_response, raw_output)
        """
        prompt = self.format_prompt(user_message, history)

        cmd = [
            self.binary_path,
            "-m", self.model_path,
            "-p", prompt,
            "-n", str(self.max_tokens),
            "-t", str(self.threads),
            "-c", str(self.ctx_size),
            "--temp", str(self.temperature),
            "--no-warmup",
            "-ngl", "0",
        ]

        # Add Floyd-Steinberg if enabled
        if self.fs_intensity > 0:
            cmd.extend(["--fs", str(self.fs_intensity)])

        # Add ordered dithering if enabled
        if self.dither_intensity > 0:
            cmd.extend(["--dither", str(self.dither_intensity)])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
            raw_output = result.stdout
            if result.returncode != 0:
                return f"[Error: {result.stderr}]", raw_output

            response = self.parse_response(raw_output)
            return response, raw_output

        except subprocess.TimeoutExpired:
            return "[Error: Generation timed out]", ""
        except Exception as e:
            return f"[Error: {str(e)}]", ""

    def generate_pulse(self, user_message: str, history: list = None) -> Tuple[str, str, str]:
        """
        Two-pass pulse generation:
        Pass 1: Scout run - conservative settings, map the terrain
        Pass 2: Informed exploration - use Pass 1's confidence as flight path
        
        Returns:
            Tuple of (pass2_response, pass1_response, comparison_info)
        """
        prompt = self.format_prompt(user_message, history)
        
        # Pass 1: Scout - conservative, find the terrain
        # Lower dither, lower temp - find what's certain vs uncertain
        pass1_dither = self.dither_intensity * 0.5  # More conservative
        pass1_temp = self.temperature * 0.7  # Cooler, more committed
        
        cmd1 = [
            self.binary_path,
            "-m", self.model_path,
            "-p", prompt,
            "-n", str(self.max_tokens),
            "-t", str(self.threads),
            "-c", str(self.ctx_size),
            "--temp", str(pass1_temp),
            "--no-warmup",
            "-ngl", "0",
            "--seed", "42",  # Fixed seed for Pass 1 reproducibility
        ]
        
        if self.fs_intensity > 0:
            cmd1.extend(["--fs", str(self.fs_intensity)])
        if pass1_dither > 0:
            cmd1.extend(["--dither", str(pass1_dither)])
        
        try:
            result1 = subprocess.run(cmd1, capture_output=True, text=True, timeout=120)
            pass1_raw = result1.stdout
            pass1_response = self.parse_response(pass1_raw)
        except Exception as e:
            return f"[Pass 1 Error: {e}]", "", ""
        
        # Pass 2: Informed exploration - higher dither, guided by Pass 1 context
        # Pass 1's output becomes context - the model now KNOWS the terrain
        pass2_dither = self.dither_intensity * 1.5  # More exploratory
        pass2_temp = self.temperature * 1.1  # Warmer, more adventurous
        
        # Build Pass 2 prompt: original + Pass 1 output as "initial thoughts"
        pass2_prompt = self.format_prompt(
            f"{user_message}\n\n[Your initial thoughts: {pass1_response}]\n\nNow, with that understanding, give your refined response:",
            history
        )
        
        cmd2 = [
            self.binary_path,
            "-m", self.model_path,
            "-p", pass2_prompt,
            "-n", str(self.max_tokens),
            "-t", str(self.threads),
            "-c", str(self.ctx_size),
            "--temp", str(pass2_temp),
            "--no-warmup",
            "-ngl", "0",
        ]
        
        if self.fs_intensity > 0:
            cmd2.extend(["--fs", str(self.fs_intensity * 0.8)])  # Slightly less F-S, more freedom
        if pass2_dither > 0:
            cmd2.extend(["--dither", str(pass2_dither)])
        
        try:
            result2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=120)
            pass2_raw = result2.stdout
            pass2_response = self.parse_response(pass2_raw)
        except Exception as e:
            return f"[Pass 2 Error: {e}]", pass1_response, ""
        
        comparison = f"Pass 1 (scout): temp={pass1_temp:.3f}, dither={pass1_dither:.3f}\n"
        comparison += f"Pass 2 (explore): temp={pass2_temp:.3f}, dither={pass2_dither:.3f}"
        
        return pass2_response, pass1_response, comparison

    def chat(self):
        """Run an interactive chat session."""
        print(f"\n{'='*60}")
        print("Floyd-Steinberg Chat")
        print(f"{'='*60}")
        print(f"Model: {self.model_path}")
        print(f"F-S Intensity: {self.fs_intensity}")
        print(f"Dither: {self.dither_intensity}")
        print(f"Temperature: {self.temperature}")
        print(f"System: {self.system_prompt}")
        print(f"{'='*60}")
        print("Type 'quit' or 'exit' to end. Type 'raw' to toggle raw output.\n")

        history = []
        show_raw = False

        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break
            except KeyboardInterrupt:
                print("\n")
                break

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit"):
                print("Goodbye!")
                break

            if user_input.lower() == "raw":
                show_raw = not show_raw
                print(f"[Raw output: {'ON' if show_raw else 'OFF'}]")
                continue

            if user_input.lower() == "clear":
                history = []
                print("[History cleared]")
                continue

            print("Assistant: ", end="", flush=True)
            response, raw = self.generate(user_input, history)
            print(response)

            if show_raw:
                print(f"\n--- Raw Output ---\n{raw}\n--- End Raw ---\n")

            # Update history
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})

            print()


def main():
    parser = argparse.ArgumentParser(
        description="Floyd-Steinberg Chat - Interactive chat with F-S dithered inference"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Path to model file"
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=0.1618,
        help="Floyd-Steinberg intensity (0.0 to disable)"
    )
    parser.add_argument(
        "--dither",
        type=float,
        default=0.0432,
        help="Ordered dithering intensity (0.0 to disable)"
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.528,
        help="Temperature"
    )
    parser.add_argument(
        "--system",
        type=str,
        default="You are a helpful assistant.",
        help="System prompt"
    )
    parser.add_argument(
        "-n", "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "-t", "--threads",
        type=int,
        default=8,
        help="Number of threads"
    )
    parser.add_argument(
        "--message",
        type=str,
        default=None,
        help="Single message mode (non-interactive)"
    )
    parser.add_argument(
        "--pulse",
        action="store_true",
        help="Enable two-pass pulse mode (Pass 1: scout, Pass 2: informed exploration)"
    )

    args = parser.parse_args()

    try:
        chat = FSChat(
            model_path=args.model,
            system_prompt=args.system,
            fs_intensity=args.fs,
            dither_intensity=args.dither,
            temperature=args.temp,
            max_tokens=args.max_tokens,
            threads=args.threads,
        )

        if args.message:
            if args.pulse:
                # Two-pass pulse mode
                pass2, pass1, info = chat.generate_pulse(args.message)
                print(f"=== PASS 1 (Scout) ===\n{pass1}\n")
                print(f"=== PASS 2 (Informed Exploration) ===\n{pass2}\n")
                print(f"--- Settings ---\n{info}")
            else:
                # Single message mode
                response, _ = chat.generate(args.message)
                print(response)
        else:
            # Interactive mode
            chat.chat()

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
