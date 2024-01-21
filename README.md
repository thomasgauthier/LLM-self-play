# Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models

This is an implementation of the paper: 
> [Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models](https://arxiv.org/abs/2401.01335) <br/>
> Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, Quanquan Gu <br/>

---

# Summary

## Abstract

> Harnessing the power of human-annotated data through Supervised Fine-Tuning (SFT) is pivotal for advancing Large Language Models (LLMs). In this paper, we delve into the prospect of growing a strong LLM out of a weak one without the need for acquiring additional human-annotated data. We propose a new fine-tuning method called Self-Play fIne-tuNing (SPIN), which starts from a supervised fine-tuned model. At the heart of SPIN lies a self-play mechanism, where the LLM refines its capability by playing against instances of itself. More specifically, the LLM generates its own training data from its previous iterations, refining its policy by discerning these self-generated responses from those obtained from human-annotated data. Our method progressively elevates the LLM from a nascent model to a formidable one, unlocking the full potential of human-annotated demonstration data for SFT. Theoretically, we prove that the global optimum to the training objective function of our method is achieved only when the LLM policy aligns with the target data distribution. Empirically, we evaluate our method on several benchmark datasets including the HuggingFace Open LLM Leaderboard, MT-Bench, and datasets from Big-Bench. Our results show that SPIN can significantly improve the LLM’s performance across a variety of benchmarks and even outperform models trained through direct preference optimization (DPO) supplemented with extra GPT-4 preference data. This sheds light on the promise of self-play, enabling the achievement of human-level performance in LLMs without the need for expert opponents.

## Algorithm

For each iteration \( t \):
- The **opponent** is the model with parameters \( \theta_t \) from the previous iteration.
- The **main player** is conceptually the role of the updated model for the current iteration, which is being trained and will have parameters \( \theta_{t+1} \) after the update.

When we move to the next iteration \( t+1 \):
- The newly updated model with parameters \( \theta_{t+1} \) becomes the opponent for this iteration.
- The main player will again be the role of this model after it is updated in the current iteration, which will result in a new set of parameters \( \theta_{t+2} \).

So, for each iteration, the same model updates its parameters and switches roles from the main player (the model being updated) to the opponent (the model generating synthetic responses) for the next iteration. This cycle continues until the training process concludes after \( T \) iterations.

To put it simply, after each training step, the same model takes on the role of the opponent for generating synthetic data for the next iteration, and then it is trained (as the main player) to update its parameters.
"""

Let's take an extensive look at the implementation and the considerations involved:

### Objective

⚠️ This repository includes additional LoRA code to use the SPIN algorithm with parameter efficient training

We are implementing a fine-tuning method for a pre-trained language model (LM), specifically GPT-Neo, using the Self-Play Fine-Tuning (SPIN) approach described in a research paper. The goal is to enhance the LM's performance by iteratively training it to distinguish between its own synthetic responses and ground truth responses.

#### Methodology - Self-Play Fine-Tuning (SPIN)
1. **Two Roles – Opponent and Main Player**: The LM takes on two roles during training:
   - **Opponent**: Generates synthetic responses based on prompts from a dataset.
   - **Main Player**: Trains to differentiate between these synthetic responses and the ground truth responses.

2. **Iterative Training Process**: The process involves multiple iterations where the model in the main player role is trained against its own outputs (synthetic responses) generated in the opponent role from the previous iteration.

3. **Low-Rank Adaptation (LoRA)**: To make this training process efficient, we utilize LoRA, a parameter-efficient fine-tuning method that adds trainable low-rank matrices to certain layers of the LM. This approach drastically reduces the number of parameters that need fine-tuning, facilitating rapid adaptation.

4. **Adapting to New Roles**: After each training iteration, the roles switch – the updated model becomes the new opponent for the next iteration.

## Implementation Details
1. **Model Setup**:
   - Utilize GPT-Neo 2.7B as the base model.
   - Implement LoRA for parameter-efficient training.

2. **LoRA Configuration**:
   - Apply LoRA to the LM's linear layers.
   - Configure LoRA with specific parameters like rank (`r`), scaling factor (`lora_alpha`), and dropout (`lora_dropout`).

3. **Training Procedure**:
   - In each iteration, first generate synthetic responses (opponent role).
   - Then, train the model (main player role) using a specialized loss function.

4. **Specialized Loss Function**:
   - Implement SPIN loss that maximizes the difference in predicted probabilities for ground truth and synthetic responses.
   - Use logistic loss to calculate this difference.

5. **Parameter-Efficient Training with LoRA**:
   - Use `LoraModel` from the PEFT library to add LoRA layers to the LM.
   - Toggle LoRA layers' trainability using `disable_adapter_layers` and `enable_adapter_layers` for switching between opponent and main player roles.

6. **Optimizer**:
   - Use AdamW optimizer, ensuring it only updates parameters that are currently set to be trainable (i.e., the LoRA parameters during the main player phase).

7. **Considerations**:
   - Memory Efficiency: By using LoRA, we efficiently manage memory usage since we don't have to duplicate the entire model for the two roles but only modify a small set of additional parameters.
   - Iterative Role Switching: Carefully manage the role switching between opponent and main player to ensure that the model correctly learns to distinguish between synthetic and real responses.
   - Dataset and Loss Function: The choice of dataset and the design of the SPIN loss function are crucial for the success of the training process.

### Conclusion
This implementation aims to improve the LM's performance by iteratively training it in a self-play manner, leveraging the efficiency of LoRA for fine-tuning. The process involves careful handling of roles, efficient use of memory, and a specific training regimen that aligns with the principles outlined in the SPIN methodology.