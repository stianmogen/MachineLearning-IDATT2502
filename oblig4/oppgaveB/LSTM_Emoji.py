import torch
import torch.nn as nn
import numpy as np


class LongShortTermMemoryModel(nn.Module):
    def __init__(self, encoding_size, emoji_size):
        super(LongShortTermMemoryModel, self).__init__()

        self.lstm = nn.LSTM(encoding_size, 128)  # 128 is the state size
        self.dense = nn.Linear(128, emoji_size)  # 128 is the state size

    def reset(self):  # Reset states prior to new input sequence
        zero_state = torch.zeros(1, 1, 128)  # Shape: (number of layers, batch size, state size)
        self.hidden_state = zero_state
        self.cell_state = zero_state

    def logits(self, x):  # x shape: (sequence length, batch size, encoding size)
        out, (self.hidden_state, self.cell_state) = self.lstm(x, (self.hidden_state, self.cell_state))
        return self.dense(out.reshape(-1, 128))

    def f(self, x):  # x shape: (sequence length, batch size, encoding size)
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x, y):  # x shape: (sequence length, batch size, encoding size), y shape: (sequence length, encoding size)
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))


index_to_char = [' ', 'h', 'a', 't', 'r', 'c', 'f', 'l', 'm', 'p', 's', 'o', 'n']
char_encodings = np.eye(len(index_to_char))
encoding_size = len(char_encodings)

emojis = {
        'hat': '\U0001F3A9',
        'cat': '\U0001F408',
        'rat': '\U0001F400',
        'flat': '\U0001F3E2',
        'matt': '\U0001F468',
        'cap': '\U0001F9E2',
        'son': '\U0001F466'
}
index_to_emoji = [emojis['hat'], emojis['rat'], emojis['cat'],
                      emojis['flat'], emojis['matt'], emojis['cap'], emojis['son']]
emoji_encodings = np.eye(len(index_to_emoji))
emoji_size = len(emoji_encodings)

x_train = torch.tensor([[[char_encodings[1]], [char_encodings[2]], [char_encodings[3]], [char_encodings[0]]],  # "hat "
                        [[char_encodings[4]], [char_encodings[2]], [char_encodings[3]], [char_encodings[0]]],  # "rat "
                        [[char_encodings[5]], [char_encodings[2]], [char_encodings[3]], [char_encodings[0]]],  # "cat "
                        [[char_encodings[6]], [char_encodings[7]], [char_encodings[2]], [char_encodings[3]]],  # "flat"
                        [[char_encodings[8]], [char_encodings[2]], [char_encodings[3]], [char_encodings[3]]],  # "matt"
                        [[char_encodings[5]], [char_encodings[2]], [char_encodings[9]], [char_encodings[0]]],  # "cap "
                        [[char_encodings[10]], [char_encodings[11]], [char_encodings[12]], [char_encodings[0]]]],  # "son "
                       dtype=torch.float)
y_train = torch.tensor([[emoji_encodings[0], emoji_encodings[0], emoji_encodings[0], emoji_encodings[0]],  # "hat "
                        [emoji_encodings[1], emoji_encodings[1], emoji_encodings[1], emoji_encodings[1]],  # "rat "
                        [emoji_encodings[2], emoji_encodings[2], emoji_encodings[2], emoji_encodings[2]],  # "cat "
                        [emoji_encodings[3], emoji_encodings[3], emoji_encodings[3], emoji_encodings[3]],  # "flat"
                        [emoji_encodings[4], emoji_encodings[4], emoji_encodings[4], emoji_encodings[4]],  # "matt"
                        [emoji_encodings[5], emoji_encodings[5], emoji_encodings[5], emoji_encodings[5]],  # "cap "
                        [emoji_encodings[6], emoji_encodings[6], emoji_encodings[6], emoji_encodings[6]]],  # "son "
                       dtype=torch.float)


model = LongShortTermMemoryModel(encoding_size, emoji_size)

def generate(string):
    model.reset()
    for i in range(len(string)):
        char_index = index_to_char.index(string[i])
        y = model.f(torch.tensor([[char_encodings[char_index]]], dtype=torch.float))
        if i == len(string) - 1:
            print(index_to_emoji[y.argmax(1)], ": ", string)

optimizer = torch.optim.RMSprop(model.parameters(), 0.0001)
for epoch in range(500):
    for i in range(x_train.size()[0]):
        model.reset()
        model.loss(x_train[i], y_train[i]).backward()
        optimizer.step()
        optimizer.zero_grad()

generate("cat")
generate("son")
generate("rat")
generate("ras")
generate("hat")
generate("rt")
generate("rhats")
generate("s")
generate("ct")
generate("chat")
generate("hcat")