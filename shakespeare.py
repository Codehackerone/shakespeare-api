import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, GRU
from tensorflow.keras.losses import sparse_categorical_crossentropy


def create_seq_targets(seq):
    input_txt = seq[:-1]
    target_txt = seq[1:]
    return input_txt, target_txt


def sparse_cat_loss(y_true, y_pred):
    return sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)


def create_model(vocab_size, embed_dim, rnn_neurons, batch_size):
    model = Sequential()
    model.add(Embedding(vocab_size, embed_dim, batch_input_shape=[batch_size, None]))
    model.add(GRU(rnn_neurons, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'))
    # Final Dense Layer to Predict
    model.add(Dense(vocab_size))
    model.compile(optimizer='adam', loss=sparse_cat_loss)
    return model


def generate_text(model, start_seed, gen_size=100, temp=1.0):
    num_generate = gen_size
    input_eval = [char_to_ind[s] for s in start_seed]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []
    temperature = temp
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(ind_to_char[predicted_id])
    return (start_seed + ''.join(text_generated))


seq_len = 120
batch_size = 128
buffer_size = 10000
embed_dim = 64
rnn_neurons = 1026
path_to_file = 'shakespeare.txt'
text = open(path_to_file, 'r').read()
vocab = sorted(set(text))
char_to_ind = {u: i for i, u in enumerate(vocab)}
ind_to_char = np.array(vocab)
encoded_text = np.array([char_to_ind[c] for c in text])
total_num_seq = len(text) // (seq_len + 1)
char_dataset = tf.data.Dataset.from_tensor_slices(encoded_text)
sequences = char_dataset.batch(seq_len + 1, drop_remainder=True)
dataset = sequences.map(create_seq_targets)
dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
vocab_size = len(vocab)
model = create_model(vocab_size, embed_dim, rnn_neurons, batch_size=1)
model.load_weights('shakespeare_gen.h5')
model.build(tf.TensorShape([1, None]))


# print(generate_text(model, "Antonio", gen_size=1000))


def generate_text_2(text, gen_size):
    return generate_text(model, text, gen_size)

# print(generate_text_2("antonio",1000))
