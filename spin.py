import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, LoraModel, get_peft_model

# Load model and tokenizer
model_checkpoint = 'EleutherAI/gpt-neo-125M'
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(model_checkpoint).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer.pad_token = tokenizer.eos_token

# Define lambda regularization parameter as per paper details
lambda_reg = 0.1

# Placeholder for the dataset loading function
dataset = [{"prompt": "Example prompt", "response": "Example response"}]

# Define LoRA configuration
lora_config = LoraConfig(
    r=128,  # rank of LoRA
    lora_alpha=256,  # scaling factor for initialization
    lora_dropout=0.05,
    bias="none",
)

# Wrap the model with LoRA layers for parameter-efficient training
peft_model = get_peft_model(model, lora_config)

# Define the compute_spin_loss function (with expected tensor shapes)
def compute_spin_loss(model_logits_gt, opponent_logits_gt, model_logits_syn, opponent_logits_syn, ground_truth_ids, synthetic_response_ids, lambda_reg):
    # Apply softmax to convert logits to probabilities
    # Shapes after softmax: [batch_size, sequence_length, vocab_size]
    model_probs_gt = torch.nn.functional.softmax(model_logits_gt, dim=-1)
    opponent_probs_gt = torch.nn.functional.softmax(opponent_logits_gt, dim=-1)
    model_probs_syn = torch.nn.functional.softmax(model_logits_syn, dim=-1)
    opponent_probs_syn = torch.nn.functional.softmax(opponent_logits_syn, dim=-1)

    # Gather log probabilities for the actual tokens in the ground truth sequence
    # [batch_size, sequence_length, vocab_size] -> [batch_size, sequence_length]
    log_model_probs_gt = torch.log(torch.gather(
        model_probs_gt, dim=2, index=ground_truth_ids.unsqueeze(-1)
    ).squeeze(-1))
    log_opponent_probs_gt = torch.log(torch.gather(
        opponent_probs_gt, dim=2, index=ground_truth_ids.unsqueeze(-1)
    ).squeeze(-1))

    # Gather log probabilities for the actual tokens in the synthetic sequence
    # [batch_size, sequence_length, vocab_size] -> [batch_size, sequence_length]
    log_model_probs_syn = torch.log(torch.gather(
        model_probs_syn, dim=2, index=synthetic_response_ids.unsqueeze(-1)
    ).squeeze(-1))
    log_opponent_probs_syn = torch.log(torch.gather(
        opponent_probs_syn, dim=2, index=synthetic_response_ids.unsqueeze(-1)
    ).squeeze(-1))

    # Calculate log probability ratios for the tokens in the sequence
    # [batch_size, sequence_length]
    log_prob_ratio_gt = log_model_probs_gt - log_opponent_probs_gt
    log_prob_ratio_syn = log_model_probs_syn - log_opponent_probs_syn

    # Sum the log probability ratios over the sequence
    # [batch_size] -> scalar
    sum_log_prob_ratio_gt = torch.sum(log_prob_ratio_gt, dim=1)
    sum_log_prob_ratio_syn = torch.sum(log_prob_ratio_syn, dim=1)

    # Calculate the combined loss term for each sequence in the batch, scaled by lambda_reg
    # [batch_size] -> scalar
    combined_loss = lambda_reg * (sum_log_prob_ratio_gt - sum_log_prob_ratio_syn)

    # Apply the logistic loss to the combined term
    # [batch_size] -> scalar
    logistic_loss = torch.log(1 + torch.exp(-combined_loss))

    # Compute the mean of the logistic loss across the batch
    # scalar
    spin_loss = logistic_loss.mean()
    return spin_loss

# Training setup
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, peft_model.parameters()), lr=5e-5)

# Training loop for T iterations
T = 5  # Set the number of iterations
for iteration in range(T):
    total_loss = 0
    
    # Disable adapter layers for the opponent model
    peft_model.disable_adapter_layers()
    
    synthetic_data = []
    opponent_logits_gt_list = []
    for data in dataset:
        prompt = data['prompt']
        # Tokenize and generate synthetic data using the opponent model
        prompt_encoding = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to(device)
        prompt_ids = prompt_encoding['input_ids']
        prompt_attention_mask = prompt_encoding['attention_mask']
        with torch.no_grad():
            peft_model.eval()  # Set model to evaluation mode
            #Generate synthetic responses using the opponent model
            synthetic_response_ids = peft_model.generate(
                prompt_ids, 
                attention_mask=prompt_attention_mask, 
                max_length=50
            )
            synthetic_data.append(synthetic_response_ids)
            
            # Calculate opponent's logits for ground truth responses
            ground_truth = data['response']
            ground_truth_encoding = tokenizer(
                ground_truth, return_tensors='pt', padding=True, truncation=True
            ).to(device)
            ground_truth_ids = ground_truth_encoding['input_ids']
            ground_truth_attention_mask = ground_truth_encoding['attention_mask']
            opponent_logits_gt = peft_model(
                input_ids=ground_truth_ids, 
                attention_mask=ground_truth_attention_mask
            ).logits
            opponent_logits_gt_list.append(opponent_logits_gt)

    # Enable adapter layers for training the main player model
    peft_model.enable_adapter_layers()
    
    # Train the main player model using the synthetic data and real responses
    peft_model.train()  # Set model to training mode
    for i, data in enumerate(dataset):
        # Tokenize ground truth response for training
        ground_truth_encoding = tokenizer(
            data['response'], return_tensors='pt', padding=True, truncation=True
        ).to(device)
        ground_truth_ids = ground_truth_encoding['input_ids']
        ground_truth_attention_mask = ground_truth_encoding['attention_mask']
        synthetic_response_ids = synthetic_data[i].to(device)
        opponent_logits_gt = opponent_logits_gt_list[i]

        # Calculate logits for ground truth and synthetic responses using the main player model
        main_player_logits_gt = peft_model(
            input_ids=ground_truth_ids, 
            attention_mask=ground_truth_attention_mask
        ).logits
        main_player_logits_syn = peft_model(
            input_ids=synthetic_response_ids
        ).logits

        # Compute logits for synthetic responses using the opponent model (disabled adapter layers)
        peft_model.disable_adapter_layers()
        opponent_logits_syn = peft_model(
            input_ids=synthetic_response_ids
        ).logits
        peft_model.enable_adapter_layers()

        # Compute the loss
        loss = compute_spin_loss(
            main_player_logits_gt, opponent_logits_gt,
            main_player_logits_syn, opponent_logits_syn,
            ground_truth_ids, synthetic_response_ids, lambda_reg
        )
        total_loss += loss.item()

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print average loss
    average_loss = total_loss / len(dataset)
    print(f"Iteration {iteration + 1}/{T}, Average Loss: {average_loss}")

# Save the final model parameters
final_model_params = peft_model.state_dict()
print("Training complete.")