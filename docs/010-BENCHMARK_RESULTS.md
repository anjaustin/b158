# Dither Sampler: Final Benchmark Results

## Test Configuration

- **Model**: BitNet-b1.58-2B-4T (ggml-model-i2_s.gguf)
- **Prompt**: "Hypothetically, might reflective recursion be a function of cognition?"
- **System**: "You are a friend."
- **Mode**: Conversation (-cnv)
- **Temperature**: 0.8 (default)
- **Repeat Penalty**: 1.2
- **Seed**: 42

## Results

### NO DITHER (Control)

> Reflective recursion refers to the ability to think about one's own thought processes. This cognitive process can involve analyzing and understanding how we reason or problem-solve.
> 
> In essence, it would mean that our brains are not just processing information but also thinking "about" these thoughts themselves - which is a complex form of metacognition (thinking about your thoughts).
> 
> So yes, reflective recursion could potentially be considered a function of cognition, as it requires the brain to process and analyze its own cognitive processes. However, research in psychology is still ongoing about this concept, particularly how it relates to metacognition, decision-making, and problem-solving strategies.
> 
> It's important to note that while reflective recursion can be seen within human cognition, whether or not other animals possess similar capabilities has been a topic of scientific inquiry but remains largely speculative.

**Characteristics:**
- More cautious/hedging language
- References ongoing research
- Shorter response

### FIXED DITHER (--dither 0.7321)

> Reflective recursion refers to the ability to think about one's own thought processes. This is often described as "thinking about thinking." It has been proposed that this cognitive ability may develop during early childhood and can evolve over time.
> 
> In terms of whether it could potentially exist in artificial intelligence or other non-biological systems, here are a few points to consider:
> 
> 1. Human Cognition: Human beings exhibit reflective recursion, which is a key aspect of metacognition. This involves being able to think about one's own mental processes and adjust them if necessary.
> 
> 2. Artificial Intelligence: Research in AI has shown that artificial systems can also be programmed with capabilities for meta-cognitive functions like reflection on their thought processes.
> 
> 3. Animal Cognition: There is some evidence that certain animals, particularly primates and dolphins, show signs of reflective behavior.

**Characteristics:**
- Structured numbered list format
- Covers AI and animals
- Longer response
- More definitive tone

### ADAPTIVE DITHER (--dither-adaptive 0.7321)

> Reflective recursion refers to the ability to think about one's own thought processes. This is often described as "thinking about thinking." It has been proposed that this cognitive ability could have evolutionary roots and potentially even serve an adaptive purpose.
> 
> 1. Evolutionary Perspective: From a biological standpoint, recursive cognition might offer several advantages:
> 
>     - Adaptive Problem Solving: Recursive thought processes could enable an individual to adaptively solve complex problems by breaking them down into simpler parts
> 
>     - Learning and Memory: By reflecting on thought patterns or learning methods, individuals can refine their understanding of information
> 
> 2. Cognitive Science Perspective:
> 
>     - Metacognition: Reflective recursion involves metacognition – thinking about one's own cognitive processes
> 
>     - Self-Reflection: It also involves self-reflection or introspection
> 
> 3. Adaptive Function: Reflective recursion could be adaptive because it allows for a constant evolution of one's cognitive strategies

**Characteristics:**
- Focuses on evolutionary/adaptive perspective
- More coherent flow
- Bullet-point structure

## Key Differences

| Aspect | No Dither | Fixed Dither | Adaptive Dither |
|--------|------------|--------------|-----------------|
| Tone | Hedging | Definitive | Analytical |
| Structure | Paragraph | Numbered lists | Bulleted |
| Topics | Psychology focus | Human/AI/Animal | Evolution/Adaptation |
| Length | Short | Medium | Medium |

## Verified Working

- ✅ `--dither N` (fixed intensity)
- ✅ `--dither-adaptive N` (spread-based adaptive intensity)
- ✅ Conversation mode (-cnv)
- ✅ Default temperature (0.8)
- ✅ Repeat penalty (1.2)

## Notes

- Temperature 0.528 causes model instability in conversation mode - use default (0.8)
- All three modes produce coherent, non-repeating output
- Dither influences topic selection and response structure

## Date

March 2026
