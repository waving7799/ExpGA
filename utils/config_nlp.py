class config():
    num_classes = {'imdb': 2, 'yahoo': 10, 'agnews': 4, 'wiki': 2, 'sst': 2}
    word_max_len = {'imdb': 256, 'yahoo': 1000, 'agnews': 150, 'wiki': 150, 'sst': 150}
    char_max_len = {'agnews': 1014}
    num_words = {'imdb': 40000, 'yahoo': 20000, 'agnews': 5000, 'wiki': 100000, 'sst': 20000}

    wordCNN_batch_size = {'imdb': 32, 'yahoo': 32, 'agnews': 32, 'wiki': 32, 'sst': 32}
    wordCNN_epochs = {'imdb': 2, 'yahoo': 10, 'agnews': 2, 'wiki': 2, 'sst': 2}

    bdLSTM_batch_size = {'imdb': 32, 'yahoo': 32, 'agnews': 64, 'wiki': 32, 'sst': 32}
    bdLSTM_epochs = {'imdb': 6, 'yahoo': 16, 'agnews': 2, 'wiki': 2, 'sst': 2}

    charCNN_batch_size = {'agnews': 128}
    charCNN_epochs = {'agnews': 4}

    LSTM_batch_size = {'imdb': 32, 'agnews': 64, 'wiki': 32, 'sst': 64}
    LSTM_epochs = {'imdb': 15, 'agnews': 30, 'wiki': 4, 'sst': 16}

    loss = {'imdb': 'binary_crossentropy', 'yahoo': 'categorical_crossentropy', 'agnews': 'categorical_crossentropy',
            'wiki': 'binary_crossentropy','sst': 'binary_crossentropy'}

    activation = {'imdb': 'sigmoid', 'yahoo': 'softmax', 'agnews': 'softmax','wiki': 'sigmoid','sst': 'sigmoid'}

    wordCNN_embedding_dims = {'imdb': 50, 'yahoo': 50, 'agnews': 50, 'wiki': 50, 'sst': 50}
    bdLSTM_embedding_dims = {'imdb': 128, 'yahoo': 128, 'agnews': 128, 'sst': 128}
    LSTM_embedding_dims = {'imdb': 100, 'agnews': 100, 'wiki': 100, 'sst': 100}
