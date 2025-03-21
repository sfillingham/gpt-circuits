import torch



# Load the Shakespeare validation data
file_path = '/Volumes/MacMini/gpt-circuits/data/shakespeare/val_000000.npy'
val_input = np.load(file_path)

#Identify the locations of the output logit of interest
new_line_indices = np.where(val_input == 10)[0]
left_cut = 0
right_cut = 0
nl_prompt_list = []
dl_prompt_list = []

for i, idx in enumerate(new_line_indices):
    right_cut = idx
    if right_cut == left_cut:
        continue
    elif right_cut - left_cut == 1:
        left_cut = right_cut
        continue
    else:
        #Grab the sequence between newline characters plus the next two characters to check if its a double line
        token_sequence = val_input[left_cut+1:right_cut+2]
        if token_sequence[-1] == 10:
            dl_prompt_list.append(tokenizer.decode_sequence(token_sequence[:-2]))
        else:
            #if len(token_sequence) >= 10:
            nl_prompt_list.append(tokenizer.decode_sequence(token_sequence[:-2]))
        left_cut = right_cut

# Select the prompts that are the correct length
sample_dl_prompt_list = []
for prompt in dl_prompt_list:
    if len(prompt) == 16:
        print(prompt)
        print(len(tokenizer.encode(prompt)))
        sample_dl_prompt_list.append(prompt)

# Generate output for the selected prompts
model_output = []
for prompt in sample_dl_prompt_list:
    output = generate(model, tokenizer, prompt, max_length=1)
    model_output.append(output)

# Put all of the prompts that return the correct token in a list
output_list = []
for output in model_output:
    if output[0][-1] == 10:# and output[0][-2]:
        count += 1
        output_list.append(output[0][:-1])
        #print(output[0][:-1])

#save the output list for use in ablation experiments
torch.save(output_list, 'double_newline_prompts.pt')
