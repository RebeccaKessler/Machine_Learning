recommended parameters:
learning rate: 0.0001 - 0.0002
epochs: 3-7
batchsize (8, 16, 32)
weigth decay = 0.01 (for now)


Combinations


{'learning_rate': 0.00019, 'num_train_epochs': 5, 'per_device_train_batch_size': 32}
Accuracy: 0.57


{'learning_rate': 0.0004, 'num_train_epochs': 6, 'per_device_train_batch_size': 32, 'weight_decay': 0.02} -> let's not focus on weight-decay yet
Accuracy: 0.57


{'learning_rate': 0.00008, 'num_train_epochs': 10, 'per_device_train_batch_size': 8, 'weight_decay': 0.06} -> let's not focus on weight-decay yet
Accuracy: 0.57


{'learning_rate': 0.0002, 'num_train_epochs': 5, 'per_device_train_batch_size': 32}
Accuracy: 0.55


{'learning_rate': 0.0002, 'num_train_epochs': 6, 'per_device_train_batch_size': 32}
Accuracy: 0.56


{'learning_rate': 0.0002, 'num_train_epochs': 7, 'per_device_train_batch_size': 32}
Accuracy: 0.55


{'learning_rate': 0.0002, 'num_train_epochs': 7, 'per_device_train_batch_size': 16}
Accuracy: 0.525


{'learning_rate': 0.00005, 'num_train_epochs': 3, 'per_device_train_batch_size': 8}
Accuracy: 0.57

{'learning_rate': 0.0005, 'num_train_epochs': 6, 'per_device_train_batch_size': 16}
Accuracy: 0.576


{'learning_rate': 18e-5, 'num_train_epochs': 5, 'per_device_train_batch_size': 32}
Accuracy: 0.566

{'learning_rate': 17e-5, 'num_train_epochs': 5, 'per_device_train_batch_size': 32}
Accuracy: 0.571

{'learning_rate': 16e-5, 'num_train_epochs': 5, 'per_device_train_batch_size': 32}
Accuracy: 5.569

{'learning_rate': 15e-5, 'num_train_epochs': 5, 'per_device_train_batch_size': 32}
Accuracy: 5.561

{'learning_rate': 20e-5, 'num_train_epochs': 5, 'per_device_train_batch_size': 16}
Accuracy: -> 0.557

{'learning_rate': 17e-5, 'num_train_epochs': 5, 'per_device_train_batch_size': 8}
Accuracy: -> 5.557

{'learning_rate': 18e-5, 'num_train_epochs': 5, 'per_device_train_batch_size': 16}
Accuracy: -> currently trying

{'learning_rate': 15e-5, 'num_train_epochs': 6, 'per_device_train_batch_size': 16}
Accuracy: 5.548

