from transformers import BertTokenizer, BertForMaskedLM, AdamW
from tqdm import tqdm  # for our progress bar
import torch

# Source: https://github.com/jamescalam/transformers/blob/main/course/training/03_mlm_training.ipynb

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertForMaskedLM.from_pretrained('bert-base-cased')

text_dir = 'dataset.txt'
with open(text_dir, 'r') as fp:
    text = fp.read().split('\n')

# First, we'll tokenize our text.
inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')

# Then we create our labels tensor by cloning the input_ids tensor.
inputs['labels'] = inputs.input_ids.detach().clone()
# inputs['attention_mask'] = inputs.attention_mask.detach().clone() SUGGESTED BY PYCHARM NOT IN EXAMPLE

# create random array of floats with equal dimensions to input_ids tensor
rand = torch.rand(inputs.input_ids.shape)
# create mask array
mask_arr = (rand < 0.15) * (inputs.input_ids != 101) * \
           (inputs.input_ids != 102) * (inputs.input_ids != 0)

# And now we take take the indices of each True value, within each individual vector.
selection = []

for i in range(inputs.input_ids.shape[0]):
    selection.append(
        torch.flatten(mask_arr[i].nonzero()).tolist()
    )

# Then apply these indices to each respective row in input_ids, assigning each of the values at these indices as 103.
for i in range(inputs.input_ids.shape[0]):
    inputs.input_ids[i, selection[i]] = 103
#  the values 103 have been assigned in the same positions as we found True values in the mask_arr tensor.

# The inputs tensors are now ready, and can we can begin setting them up to be fed into our model during training.
# We create a PyTorch dataset from our data.

class RechtDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)


dataset = RechtDataset(inputs)
# And initialize the dataloader, which we'll be using to load our data into the model during training.
loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

# Now we can move onto setting up the training loop. First we setup GPU/CPU usage.
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# and move our model over to the selected device
model.to(device)
# Activate the training mode of our model, and initialize our optimizer
# (Adam with weighted decay - reduces chance of overfitting).
# activate training mode
model.train()
# initialize optimizer
optim = AdamW(model.parameters(), lr=5e-5)

# Now we can move onto the training loop, we'll train for two epochs (change epochs to modify this).
epochs = 2

for epoch in range(epochs):
    # setup loop with TQDM and dataloader
    loop = tqdm(loader, leave=True)
    for batch in loop:
        # initialize calculated gradients (from prev step)
        optim.zero_grad()
        # pull all tensor batches required for training
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        # process
        outputs = model(input_ids, attention_mask=attention_mask,
                        labels=labels)
        # extract loss
        loss = outputs.loss
        # calculate loss for every parameter that needs grad update
        loss.backward()
        # update parameters
        optim.step()
        # print relevant info to progress bar
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())