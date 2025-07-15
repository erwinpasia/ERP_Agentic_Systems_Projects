## Table of Contents

### Day 1: Foundational LLMs and Text Generation
*   Deep Dive into Large Language Models (LLMs) and Text Generation
*   Introduction
*   Transformer Architecture Foundation
*   Input Processing for Transformers
*   Self-Attention Mechanism
*   Multi-Head Attention
*   Layer Normalization & Residual Connections
*   Feed-Forward Network Layer
*   Decoder-Only Architecture
*   Mixture of Experts (MoE)
*   Evolution of LLMs - Timeline & Key Models
*   LLM Training Overview
*   Fine-tuning Techniques
*   Parameter-Efficient Fine-Tuning (PEFT)
*   Prompt Engineering
*   Sampling Techniques
*   Evaluating LLMs
*   Inference Acceleration
*   Applications of LLMs
*   Conclusion and Future

### Day 1: Prompt Engineering Techniques
*   Prompt Engineering Techniques for Large Language Models
*   Introduction to Prompt Engineering
*   Configuring Model Output
*   Prompt Engineering Techniques
*   Advanced Reasoning Techniques
*   Code Prompting Applications
*   Best Practices for Prompt Engineering
*   Conclusion and Future Considerations

*   ---

# Day 1: Foundational LLMs and Text Generation

## Deep Dive into Large Language Models (LLMs) and Text Generation

### Introduction
Large Language Models (LLMs) represent a significant transformation in artificial intelligence, fundamentally altering how we interact with information and technology. These advanced AI systems, typically implemented as deep neural networks, excel at processing, understanding, and generating text that closely resembles human language. LLMs are trained on vast datasets of text, enabling them to discern complex language patterns and perform diverse tasks such as machine translation, creative writing, question answering, text summarization, and reasoning. This chapter delves into the architectural history of LLMs, fine-tuning methodologies, efficient training techniques, inference acceleration strategies, diverse applications, and illustrative code examples, providing a comprehensive understanding up to early 2025.

### Transformer Architecture Foundation
The cornerstone of modern LLMs, the Transformer architecture, emerged from a Google translation project in 2017. Its original design featured an encoder-decoder structure. The encoder processed the input text (e.g., a sentence in French) to create a meaningful representation, which the decoder then used to generate the output (e.g., the English translation) sequentially, token by token. A token can represent a complete word (like "cat") or a sub-word unit (like "pre" in "prefix").

### Input Processing for Transformers
Before processing, input text undergoes tokenization, where it is broken down into tokens based on a predefined vocabulary. Each token is then converted into a dense vector, known as an embedding, which captures its semantic meaning. Since Transformers process tokens simultaneously rather than sequentially, positional encoding is added to the embeddings. This mechanism, which can be sinusoidal or learned, preserves the original sequence information of the tokens. The choice of positional encoding can influence the model's ability to comprehend longer sentences effectively.

### Self-Attention Mechanism
The self-attention mechanism is a critical component that allows the Transformer model to weigh the importance of different words within a sentence when processing a specific word. It enables the model to understand contextual relationships, such as identifying the antecedent of a pronoun (e.g., recognizing that "it" refers to "tiger" in "The thirsty tiger drank because it was hot").

This mechanism operates using three types of vectors derived from each token's embedding:
*   **Query (Q):** Represents the current word seeking relevant context.
*   **Key (K):** Acts as a label for each word, indicating its semantic content.
*   **Value (V):** Contains the actual information or meaning the word carries.

The process involves calculating scores based on the similarity between the Query vector of one word and the Key vectors of all other words in the sentence. These scores are normalized into attention weights, signifying how much focus the current word should place on other words. Finally, a weighted sum of all Value vectors, guided by the attention weights, produces a contextually rich representation for each word. This entire comparison and calculation process is efficiently parallelized using matrix operations for Q, K, and V. The ability to simultaneously process these relationships across the entire input sequence is a key factor in the Transformer's success in capturing nuanced meaning, particularly over long distances within the text.

### Multi-Head Attention
To further enhance the model's understanding, multi-head attention executes the self-attention process multiple times in parallel, each time using different, learned Q, K, and V matrices. Each "head" can focus on distinct types of relationships within the text, such as grammatical dependencies or semantic similarities. By integrating the perspectives from these multiple heads, the model achieves a more comprehensive and deeper understanding of the input text.

### Layer Normalization & Residual Connections
Training deep neural networks like Transformers can be challenging. Layer normalization helps stabilize the learning process by normalizing the inputs across the features for each layer, leading to faster training and improved performance. Residual connections, or skip connections, provide another crucial stabilization technique. They allow the original input of a layer to bypass the layer's transformations and be added directly to its output. This helps mitigate the vanishing gradient problem, enabling the network to train deeper architectures effectively and retain information learned in earlier layers.

### Feed-Forward Network Layer
Following the attention layers, each token's representation is processed independently by a feed-forward network. This network typically consists of two linear transformations separated by a non-linear activation function, such as ReLU (Rectified Linear Unit) or GeLU (Gaussian Error Linear Unit). This component further refines the token representations and enhances the model's overall capacity to learn complex patterns.

### Decoder-Only Architecture
While the original Transformer had both an encoder and a decoder, many modern LLMs, particularly those focused on text generation tasks like writing articles or engaging in conversation, employ a decoder-only architecture. These models utilize masked self-attention, where each token can only attend to preceding tokens in the sequence. This ensures that the model generates text sequentially, predicting the next token based solely on the context that comes before it. This simpler design is highly effective for generative tasks as it focuses directly on producing coherent and contextually appropriate output token by token.

### Mixture of Experts (MoE)
The Mixture of Experts (MoE) architecture offers a strategy for building significantly larger and more capable models while managing computational costs efficiently. Instead of having one massive, dense network, an MoE model comprises multiple smaller, specialized "expert" submodels. A "gating network" directs the input tokens to only a relevant subset of these experts for processing. This selective activation means that only a fraction of the model's total parameters are used for any given input, drastically reducing the computational load during both training and inference compared to a dense model of equivalent size.

### Evolution of LLMs - Timeline & Key Models
The field of LLMs has evolved rapidly since the introduction of the Transformer:
*   **GPT-1 (2018):** A decoder-only model pioneering unsupervised pre-training but sometimes producing repetitive text.
*   **BERT (2018):** An encoder-only model focused on language understanding, trained using masked language modeling and next sentence prediction.
*   **GPT-2 (2019):** A scaled-up version of GPT-1 demonstrating improved coherence and early zero-shot learning capabilities (performing tasks without specific examples).
*   **GPT-3 Family (2020+):** Marked by massive scale (billions of parameters), few-shot learning (learning from a small number of examples), instruction tuning (InstructGPT), code generation (GPT-3.5), multimodality (GPT-4 handling images and text), and large context windows.
*   **Lambda (2021):** Focused on improving natural conversation capabilities.
*   **Gopher (2021):** Emphasized the importance of high-quality data and optimization, highlighting that simply increasing model size doesn't guarantee better performance across all tasks.
*   **GLAM:** Utilized the Mixture of Experts (MoE) architecture.
*   **Chinchilla (2022):** Demonstrated the concept of compute-optimal scaling, showing the crucial relationship between model size and the amount of training data.
*   **PaLM & PaLM 2 (2022/2023):** Achieved strong benchmark performance, featured the Pathway system, and showed improved reasoning, coding, and mathematical abilities.
*   **Gemini:** Natively multimodal, optimized for TPUs, utilized MoE, offered in various sizes (Ultra, Pro, Nano, Flash), and featured very large context windows (1.5 Pro).
*   **Open Source Models:** A growing ecosystem including Gemma/Gemma 2, the Llama family (1, 2, 3, 3.1), Mixtral (MoE), 01 (reasoning-focused), DeepSeek-R1 (reasoning), Qwen, Yi, and Grok, each with varying capabilities and licensing considerations.

### LLM Training Overview
Training LLMs typically involves two main stages:
*   **Pre-training:** An unsupervised phase where the model learns general language patterns, grammar, and world knowledge by processing massive amounts of unlabeled text data from the internet and digitized books.
*   **Fine-tuning:** A subsequent phase where the pre-trained model is further trained on smaller, more specific datasets, often labeled, to adapt its capabilities for particular tasks or domains (e.g., medical text analysis, customer service).

### Fine-tuning Techniques
Several techniques exist to adapt pre-trained models:
*   **Supervised Fine-Tuning (SFT):** Training the model on curated datasets consisting of prompt-response pairs relevant to the target task.
*   **Reinforcement Learning from Human Feedback (RLHF):** A multi-step process involving training a separate "reward model" based on human preferences for different model outputs, and then using reinforcement learning algorithms to optimize the LLM to generate outputs that maximize the reward score, aligning it better with human expectations. Related techniques like RLAIF (RL from AI Feedback) and DPO (Direct Preference Optimization) offer alternative approaches to alignment.

### Parameter-Efficient Fine-Tuning (PEFT)
Training all the parameters of extremely large LLMs can be computationally expensive. PEFT methods address this by fine-tuning only a small subset of the model's parameters while keeping the majority of the pre-trained weights frozen. This significantly reduces the computational resources and time required for adaptation. Common PEFT techniques include:
*   **Adapters:** Inserting small, trainable modules between existing layers of the frozen pre-trained model.
*   **LoRA (Low-Rank Adaptation):** Introducing trainable low-rank matrices into layers to approximate the weight updates.
*   **QLoRA:** A memory-efficient version of LoRA that uses quantization.
*   **Soft Prompting (Prompt Tuning):** Keeping the model weights frozen and instead learning specific "soft prompt" embeddings that are prepended to the input sequence to guide the model's behavior.

### Prompt Engineering
Prompt engineering is the practice of carefully designing the input text (the prompt) given to an LLM to elicit the desired output. The way a prompt is phrased significantly influences the model's response. Key techniques include:
*   **Zero-shot prompting:** Providing a direct instruction or question without any examples.
*   **Few-shot prompting:** Including a small number of examples (input-output pairs) within the prompt to demonstrate the desired task format or style.
*   **Chain-of-Thought (CoT) prompting:** Structuring the prompt to encourage the model to break down a problem into intermediate reasoning steps before providing the final answer, often improving performance on complex reasoning tasks.

### Sampling Techniques
LLMs generate text token by token, predicting the probability distribution for the next token at each step. Sampling techniques determine how the next token is selected from this distribution:
*   **Greedy Search:** Always selects the single most probable next token. It's fast but can lead to repetitive or deterministic output.
*   **Random Sampling (with Temperature):** Introduces randomness by sampling from the probability distribution. A "temperature" parameter controls the degree of randomness: higher temperatures increase creativity but also the risk of nonsensical output, while lower temperatures make the output more focused and predictable.
*   **Top-K Sampling:** Restricts the selection to the K most probable tokens, reducing the chance of picking very unlikely tokens.
*   **Top-P (Nucleus) Sampling:** Selects from the smallest set of tokens whose cumulative probability exceeds a threshold P. This provides a dynamic alternative to Top-K.
*   **Best-of-N Sampling:** Generates N candidate responses and uses a scoring mechanism (potentially another model or specific criteria) to select the best one.

### Evaluating LLMs
Evaluating the performance of LLMs is complex due to the subjective nature of language and the wide range of tasks they can perform. Traditional NLP metrics are often insufficient. A comprehensive evaluation framework considers:
*   **Task-specific data:** Using datasets that reflect real-world usage scenarios and user interactions relevant to the intended application.
*   **System-level consideration:** Evaluating the LLM as part of the larger system it integrates with (e.g., in a Retrieval Augmented Generation system).
*   **Defining "good":** Establishing clear metrics based on desired qualities like accuracy, helpfulness, factual correctness, coherence, fluency, creativity, and adherence to specific styles or constraints.

Methods for evaluation include:
*   **Quantitative metrics:** Using metrics like BLEU and ROUGE (often used in translation and summarization) to compare model output against reference "ground truth" answers, although these may not fully capture quality.
*   **Human evaluation:** Employing human raters to provide nuanced judgments on aspects like fluency, coherence, relevance, and overall quality, often considered the gold standard but expensive and time-consuming.
*   **LLM-powered evaluators (auto-evaluators):** Using another capable LLM to assess the output of the model being evaluated, based on predefined criteria or rubrics. This approach requires careful calibration and validation.
*   **Advanced approaches:** Breaking down complex tasks into subtasks and using detailed rubrics with multiple criteria, especially crucial for multimodal models handling diverse data types.

### Inference Acceleration
Making LLM inference (the process of generating responses) faster and more computationally efficient is critical for real-world deployment. This involves balancing trade-offs between response quality, speed (latency), cost, and the number of requests handled concurrently (throughput). Techniques include:
*   **Output Approximating Methods:**
    *   **Quantization:** Reducing the numerical precision of the model's weights and activations (e.g., from 32-bit floats to 8-bit integers), decreasing memory usage and potentially speeding up computation, sometimes with a minor impact on quality.
    *   **Distillation:** Training a smaller "student" model to mimic the behavior of a larger, more capable "teacher" model.
*   **Output Preserving Methods:**
    *   **FlashAttention:** An optimized implementation of the attention mechanism that reduces memory reads/writes, speeding up calculations.
    *   **Prefix Caching (KV Caching):** Reusing the computed Key and Value vectors for the initial part of the input (prompt) when generating subsequent tokens, avoiding redundant calculations.
    *   **Speculative Decoding:** Using a smaller, faster "draft" model to generate candidate token sequences, which are then efficiently verified or corrected by the larger main model.
    *   **Batching:** Processing multiple user requests simultaneously to improve hardware utilization and throughput.
    *   **Parallelization:** Distributing the model's computation across multiple processing units (like GPUs or TPUs).

### Applications of LLMs
LLMs have demonstrated remarkable capabilities across a wide spectrum of applications:
*   **Code/Math Assistance:** Generating, completing, refactoring, debugging, translating, and documenting code; explaining complex codebases; solving mathematical problems.
*   **Machine Translation:** Producing more fluent, accurate, and context-aware translations between languages.
*   **Summarization:** Condensing lengthy documents or articles into concise summaries highlighting key information.
*   **Question Answering (RAG):** Enhancing question-answering systems by retrieving relevant information from external knowledge bases before generating an answer (Retrieval Augmented Generation).
*   **Chatbots & Conversational AI:** Enabling more natural, engaging, and humanlike interactions in customer service, virtual assistants, and entertainment.
*   **Content Creation:** Assisting with or autonomously generating various forms of creative text, including articles, marketing copy, scripts, poems, and emails.
*   **Natural Language Inference:** Performing tasks like sentiment analysis, analyzing legal documents for specific clauses, assisting in medical diagnosis by processing clinical notes, and understanding customer feedback.
*   **Text Classification:** Categorizing text for applications like spam detection, news topic classification, and intent recognition.
*   **LLM Evaluation:** Serving as automated evaluators for assessing the quality of other AI models.
*   **Text Analysis:** Extracting insights, identifying trends, and summarizing opinions from large volumes of unstructured text data.
*   **Multimodal Applications:** Processing and generating content that combines text with other modalities like images, audio, and video.

### Conclusion and Future
This exploration covered the foundational Transformer architecture, the rapid evolution of LLMs, techniques for training and fine-tuning, methods for evaluation and optimization, and the expanding range of applications. The field continues to advance at an accelerated pace, prompting questions about the novel applications future LLM generations will enable and the ethical, technical, and societal challenges that need addressing for responsible development and deployment.

---

# Day 1: Prompt Engineering Techniques

## Prompt Engineering Techniques for Large Language Models

### Introduction to Prompt Engineering
Prompt engineering is the crucial skill of crafting effective inputs to guide Large Language Models (LLMs) toward desired outputs. While basic prompting is accessible to anyone, achieving specific, reliable, and high-quality results, particularly in data-intensive environments like Kaggle competitions, requires a deeper understanding of prompting techniques. This section aims to equip users, especially those on platforms like Kaggle, with practical methods to leverage LLMs for enhancing coding, data analysis, and problem-solving capabilities. We will cover fundamental concepts through advanced strategies like Chain of Thought and ReAct, tailored for competitive data science challenges.

### Configuring Model Output
Successfully utilizing LLMs involves not only crafting the input prompt but also configuring the model's generation parameters, as both influence the final output.
*   **Output Length:** The maximum number of tokens the model generates directly impacts processing time and computational cost. This is particularly relevant for platforms with output limits or resource constraints. Precise prompts are needed for concise responses, especially when using iterative techniques like ReAct where multiple model calls might occur.
*   **Sampling Controls:** These parameters govern the randomness and creativity of the model's output.
    *   **Temperature:** Controls the randomness of token selection. Lower temperatures (e.g., 0.1) produce more deterministic and focused outputs, ideal for tasks requiring specific formats or factual accuracy like generating syntactically correct code. Higher temperatures (e.g., 0.9) encourage more diversity and creativity, useful for brainstorming novel approaches or generating varied text formats.
    *   **Top-K and Top-P (Nucleus Sampling):** These parameters further refine the token selection process by limiting the pool of candidate tokens considered at each step. Top-K restricts the choice to the K most probable tokens, while Top-P selects from the smallest set of tokens whose cumulative probability exceeds a threshold P. Experimentation is key, as optimal settings vary by task. Combining these controls can fine-tune the balance between coherence and creativity.
*   **Repetition Loop Bug:** A potential issue where the model gets stuck repeating words or phrases. Careful tuning of temperature, Top-K, and Top-P can help mitigate this behavior.

**Recommendations for Kaggle:**
*   For coherent yet creative results (e.g., exploring feature engineering ideas): Temperature ~0.2, Top-P ~0.95, Top-K ~30.
*   For highly creative output (e.g., generating diverse synthetic data examples): Temperature ~0.9, Top-P ~0.99, Top-K ~40.
*   For factual accuracy (e.g., summarizing technical documentation): Temperature ~0.1, Top-P ~0.9, Top-K ~20.
*   For tasks with a single correct answer (e.g., specific code translation): Temperature 0.

### Prompt Engineering Techniques
Clear and well-structured prompts are fundamental for obtaining accurate and relevant predictions from LLMs.
*   **General Prompting (Zero-Shot Prompting):** This involves providing the task description directly without including examples. It relies on the model's pre-trained knowledge and is often effective for straightforward tasks like generating simple code snippets based on natural language descriptions.
*   **Documenting Prompts:** Systematically recording prompts and their resulting outputs is crucial, especially in competitive settings. This practice helps track effective strategies, identify patterns in model behavior, and facilitates iterative improvement and debugging.
*   **One-Shot and Few-Shot Prompting:** These techniques involve including one (one-shot) or a few (few-shot) examples of the desired input-output behavior directly within the prompt. This guides the model more explicitly on the expected format, style, and task execution. The quality and relevance of the examples are paramount; poorly chosen or misleading examples can confuse the model. Including examples covering potential edge cases can further improve robustness.
*   **System, Role, and Contextual Prompting:** These advanced techniques provide meta-instructions to shape the model's behavior more globally.
    *   **System Prompting:** Sets the overall context, purpose, or constraints for the interaction (e.g., "You are a helpful assistant specializing in Python data analysis").
    *   **Role Prompting:** Assigns a specific persona or identity to the LLM (e.g., "Act as a senior data scientist reviewing code"), influencing the tone, style, and focus of its responses.
    *   **Contextual Prompting:** Provides specific background information, data snippets, or situational details relevant to the immediate task.
*   **Step-Back Prompting:** Encourages the model to first consider the broader context or underlying principles related to a specific question before providing a direct answer. This can lead to more insightful, well-reasoned, and comprehensive outputs.

### Advanced Reasoning Techniques
For complex problems requiring multi-step logic or exploration, advanced prompting frameworks can enhance LLM reasoning capabilities.
*   **Chain of Thought (CoT) Prompting:** This technique guides the model to articulate its reasoning process step-by-step before arriving at the final answer. It involves providing examples in the prompt that demonstrate this thinking pattern. CoT improves performance on tasks requiring arithmetic, common sense, or symbolic reasoning, while also increasing the transparency and interpretability of the model's output.
*   **Self-Consistency:** An enhancement to CoT where the model generates multiple reasoning paths (multiple CoT outputs) for the same prompt, often using a non-zero temperature. The final answer is determined by a majority vote across the different paths, improving robustness and reliability by mitigating the impact of any single flawed reasoning chain.
*   **Tree of Thoughts (ToT):** Extends CoT by allowing the model to explore multiple reasoning paths concurrently, forming a tree structure. The model can evaluate the promise of intermediate thoughts and backtrack or explore different branches as needed. This is particularly suited for complex problems with large search spaces or requiring significant exploration.
*   **ReAct (Reason and Act):** A framework that interleaves reasoning steps (thought) with actions (interacting with external tools or environments). The model generates a thought about what to do next, then an action (e.g., calling an API, running code, searching a database), observes the result of the action, and then uses that observation to inform the next thought-action cycle. This enables agents to dynamically gather information and interact with the world to solve problems.
*   **Automatic Prompt Engineering (APE):** Techniques where LLMs themselves are used to generate or refine prompts for specific tasks, automating parts of the prompt engineering process.

### Code Prompting Applications
LLMs are powerful tools for various coding-related tasks:
*   **Code Generation:** Generating code snippets or entire functions based on natural language descriptions. While this can significantly speed up development, the generated code must be carefully reviewed, tested, and potentially refined to ensure correctness, efficiency, and security.
*   **Explaining Code:** Providing natural language explanations of complex or unfamiliar code segments, aiding understanding, collaboration, and learning.
*   **Translating Code:** Converting code between different programming languages. This can be useful when adapting algorithms or libraries, but the translated code requires thorough verification and testing.
*   **Debugging and Reviewing Code:** Identifying potential errors, suggesting improvements, refactoring code for clarity or efficiency, and checking for adherence to style guides.
*   **Multimodal Prompting:** An emerging area involving prompts that include non-textual inputs like images or diagrams alongside text instructions, potentially relevant for analyzing visualizations or data presented graphically in future Kaggle challenges.

### Best Practices for Prompt Engineering
To maximize the effectiveness of LLMs, consider these best practices:
*   **Provide Examples:** Use one-shot or few-shot prompting when possible to clearly demonstrate the desired output format and task execution.
*   **Simplicity and Clarity:** Design prompts that are easy for the model (and humans) to understand. Avoid ambiguity.
*   **Be Specific:** Clearly define the desired output, including format, length, tone, and any constraints.
*   **Use Positive Instructions:** Frame requests positively (e.g., "Generate Python code that does X") rather than negatively (e.g., "Don't use loops").
*   **Control Output Length:** Use parameters like max_tokens to manage response length, costs, and adhere to platform limits.
*   **Dynamic Prompts:** Use variables or placeholders in prompt templates to easily adapt prompts for different inputs, datasets, or tasks, enhancing reusability.
*   **Experimentation:** Try different phrasing, formats, examples, and model parameters to discover what works best for your specific task and model.
*   **Collaboration:** Share successful prompts and strategies with peers to accelerate learning and innovation within the community.
*   **Documentation:** Keep records of prompt attempts, configurations, and results to track progress, facilitate debugging, and build a knowledge base of effective techniques.
*   **Adapt to Model Updates:** Be aware that LLM performance and behavior can change with new model versions; prompts may need adjustments.
*   **Structured Output Formats:** For Kaggle, experiment with prompts that explicitly request output in specific structured formats (e.g., CSV, JSON) suitable for submission or further processing.
*   **Reasoning First, Then Answer:** For logical tasks, structure prompts (especially with CoT) to output the reasoning steps before the final answer. Setting temperature to zero can enhance consistency for the final answer extraction.

### Conclusion and Future Considerations
Mastering prompt engineering is becoming an essential skill for effectively leveraging LLMs, offering a significant advantage in competitive environments like Kaggle. As models continue to evolve, staying updated on new capabilities, techniques, and best practices is crucial. A mindset of continuous experimentation, iteration, adaptation, and learning is key to pushing the boundaries of what can be achieved with these powerful tools.

---
### References:
All "full versions" of Google Whitepapers covered during the training can be found on the links below:
*   [https://www.kaggle.com/whitepaper-foundational-llm-and-text-generation](https://www.kaggle.com/whitepaper-foundational-llm-and-text-generation)
*   [https://www.kaggle.com/whitepaper-prompt-engineering](https://www.kaggle.com/whitepaper-prompt-engineering)
*   [https://www.kaggle.com/whitepaper-embeddings-and-vector-stores](https://www.kaggle.com/whitepaper-embeddings-and-vector-stores)
*   [https://www.kaggle.com/whitepaper-agents](https://www.kaggle.com/whitepaper-agents)
*   [https://www.kaggle.com/whitepaper-agent-companion](https://www.kaggle.com/whitepaper-agent-companion)
*   [https://www.kaggle.com/whitepaper-solving-domains-specific-problems-using-llms](https://www.kaggle.com/whitepaper-solving-domains-specific-problems-using-llms)
*   [https://www.kaggle.com/whitepaper-operationalizing-generative-ai-on-vertex-ai-using-mlops](https://www.kaggle.com/whitepaper-operationalizing-generative-ai-on-vertex-ai-using-mlops)
