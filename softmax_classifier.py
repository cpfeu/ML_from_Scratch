import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from collections import Counter


class DataLoader:

    @staticmethod
    def load_iris_dataset():

        # load train set and labels
        with open('./datasets/iris-train.txt', 'r') as file:
            content = file.read().split('\n')[:-1]
        X_train = np.asarray([[line.split(' ')[1], line.split(' ')[2]] for line in content]).astype(float)
        y_train = np.asarray([line.split(' ')[0] for line in content]).astype(int)

        # subtract mean of training features from train set
        X_train_norm = X_train - np.mean(X_train, axis=0)

        # append 1 to each x for bias
        X_train_norm = np.asarray([[x[0], x[1], 1] for x in X_train_norm])

        # transform labels to one-hot-encoding labels
        y_train_one_hot_enc = []
        for label in y_train:
            y_train_one_hot_enc.append(np.eye(3)[label - 1])
        y_train_one_hot_enc = np.array(y_train_one_hot_enc)

        # load test set and labels
        with open('./datasets/iris-test.txt', 'r') as file:
            content = file.read().split('\n')[:-1]
        X_test = np.asarray([[line.split(' ')[1], line.split(' ')[2]] for line in content]).astype(float)
        y_test = np.asarray([line.split(' ')[0] for line in content]).astype(int)

        # subtract mean of training features from test set
        X_test_norm = X_test - np.mean(X_train, axis=0)

        # append 1 to each x for bias
        X_test_norm = np.asarray([[x[0], x[1], 1] for x in X_test_norm])

        # transform labels to one-hot-encoding labels
        y_test_one_hot_enc = []
        for label in y_test:
            y_test_one_hot_enc.append(np.eye(3)[label - 1])
        y_test_one_hot_enc = np.array(y_test_one_hot_enc)

        return X_train_norm, y_train_one_hot_enc, X_test_norm, y_test_one_hot_enc


class SoftmaxClassifier:
    X_train, y_train, X_test, y_test = DataLoader().load_iris_dataset()

    def __init__(self,
                 model_name,
                 num_classes,
                 dimension_data_point,
                 learning_rate,
                 momentum_rate,
                 mini_batch_size,
                 epochs,
                 weight_decay_factor):
        self.model_name = model_name
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.previous_gradient = np.zeros(shape=(num_classes, dimension_data_point))
        self.mini_batch_size = mini_batch_size
        self.epochs = epochs
        self.weight_decay_factor = weight_decay_factor
        self.weight_matrix = np.random.rand(num_classes, dimension_data_point)

        self.train_loss_history = None
        self.train_acc_history = None
        self.test_loss_history = None
        self.test_acc_history = None

    def train(self):

        training_start_time = datetime.now()
        print('Training start time:', training_start_time)

        # train and test data history
        self.train_loss_history = []
        self.train_acc_history = []
        self.test_loss_history = []
        self.test_acc_history = []

        for epoch in range(self.epochs):

            # ======= EPOCH START =======

            print('Epoch:', epoch)

            if (epoch > 0) and (epoch % 100 == 0):
                self.learning_rate = self.learning_rate / 2

            # prepare training data for epoch
            data_shuffled, labels_shuffled = self.shuffle_data(self.X_train, self.y_train)
            data_mini_batches, labels_mini_batches = self.create_mini_batches(data_shuffled, labels_shuffled,
                                                                              self.mini_batch_size)

            # initialize data structures to store loss and accuracy of each mini batch
            mini_batch_loss_list = []
            mini_batch_per_class_acc_lists_dict = {k: [] for k in range(self.num_classes)}

            for data_mini_batch, labels_mini_batch in zip(data_mini_batches, labels_mini_batches):

                # ======= MINI BATCH START =======

                softmax_cross_entropy_loss, predictions, targets = self.pass_through_network(data_mini_batch,
                                                                                             labels_mini_batch,
                                                                                             self.weight_matrix,
                                                                                             self.weight_decay_factor)

                # add to mini batch loss list of respective epoch
                mini_batch_loss_list.append(softmax_cross_entropy_loss)

                # get per class accuracies for this mini batch and store in dict
                for k in range(self.num_classes):
                    mini_batch_per_class_acc_lists_dict.get(k).append(self.get_class_accuracy(k,
                                                                                              predictions,
                                                                                              targets))

                # calculate gradient
                gradient = self.weight_decay_factor * self.weight_matrix + np.dot((predictions - targets).T,
                                                                                  data_mini_batch)
                momentum_corrected_gradient = (self.momentum_rate * self.previous_gradient) + \
                                              (1 - self.momentum_rate) * gradient

                # store gradient
                self.previous_gradient = momentum_corrected_gradient

                # update weight matrix
                self.weight_matrix = self.weight_matrix - self.learning_rate * momentum_corrected_gradient

                # =======MINI BATCH END=======

            # add average mini batch loss of this epoch to loss history list
            self.train_loss_history.append(np.mean(mini_batch_loss_list))
            print('Average Train Epoch Loss:', np.mean(mini_batch_loss_list))

            # add mean per class accuracy of this epoch to accuracy history list
            per_class_mean_acc_list = [np.mean(x) for x in mini_batch_per_class_acc_lists_dict.values()]
            self.train_acc_history.append(np.mean(per_class_mean_acc_list))
            print('Average Train Epoch Accuracy:', np.mean(per_class_mean_acc_list))

            # ======= TEST SET EVALUATION START =======

            # test set evaluation
            X_test_shuffled, y_test_shuffled = self.shuffle_data(self.X_test, self.y_test)
            test_set_cross_entropy_loss, test_predictions, test_targets = \
                self.pass_through_network(X_test_shuffled, y_test_shuffled,
                                          self.weight_matrix, self.weight_decay_factor)

            # add test set loss of this epoch to loss history list
            self.test_loss_history.append(test_set_cross_entropy_loss)
            print('Test Epoch Loss:', test_set_cross_entropy_loss)

            # add mean per class accuracy of this epoch to accuracy list
            per_class_acc_list = []
            for k in range(self.num_classes):
                per_class_acc_list.append(self.get_class_accuracy(k, test_predictions, test_targets))
            self.test_acc_history.append(np.mean(per_class_acc_list))
            print('Test Epoch Accuracy:', np.mean(per_class_acc_list))

            # ======= TEST SET EVALUATION END =======

            # ======= EPOCH END =======

        training_end_time = datetime.now()
        print('Training end time:', training_end_time)
        print('Training took', (training_end_time - training_start_time).total_seconds(), 'seconds to run.')

    def plot_train_history(self):

        # set up figure
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 8))

        # plot left figure with loss
        ax1.plot(list(range(self.epochs)), self.train_loss_history, label='Train')
        ax1.plot(list(range(self.epochs)), self.test_loss_history, label='Test')
        ax1.set_xlabel('Epochs')
        ax1.set_title('Cross Entropy Loss')
        ax1.legend()

        # plot right figure with accuracy
        ax2.plot(list(range(self.epochs)), self.train_acc_history, label='Train')
        ax2.plot(list(range(self.epochs)), self.test_acc_history, label='Test')
        ax2.set_xlabel('Epochs')
        ax2.set_title('Accuracy')
        ax2.legend()

        # plot figure
        plt.show()

    def plot_decision_boundaries(self):

        # set up figure
        plt.figure()
        ax = plt.axes()

        # plot decision boundaries
        x = np.linspace(-1, 1, 1000)
        ax.plot(x, (-(self.weight_matrix[0][2] + self.weight_matrix[0][0] * x) /
                    self.weight_matrix[0][1]), color='black')
        ax.plot(x, (-(self.weight_matrix[1][2] + self.weight_matrix[1][0] * x) /
                    self.weight_matrix[1][1]), color='black')
        ax.plot(x, (-(self.weight_matrix[2][2] + self.weight_matrix[2][0] * x) /
                    self.weight_matrix[2][1]), color='black')

        # plot data points
        x_cors_k1 = [data_point[0] for data_point, label in zip(self.X_train, self.y_train) if np.argmax(label) == 0]
        y_cors_k1 = [data_point[1] for data_point, label in zip(self.X_train, self.y_train) if np.argmax(label) == 0]
        ax.scatter(x_cors_k1, y_cors_k1, color='green', label='Class 1')

        x_cors_k2 = [data_point[0] for data_point, label in zip(self.X_train, self.y_train) if np.argmax(label) == 1]
        y_cors_k2 = [data_point[1] for data_point, label in zip(self.X_train, self.y_train) if np.argmax(label) == 1]
        ax.scatter(x_cors_k2, y_cors_k2, color='red', label='Class 2')

        x_cors_k3 = [data_point[0] for data_point, label in zip(self.X_train, self.y_train) if np.argmax(label) == 2]
        y_cors_k3 = [data_point[1] for data_point, label in zip(self.X_train, self.y_train) if np.argmax(label) == 2]
        ax.scatter(x_cors_k3, y_cors_k3, color='blue', label='Class 3')

        # plot figure
        plt.title('Decision Boundaries of Softmax Classifier')
        plt.legend()
        plt.show()

    # get class accuracies
    @classmethod
    def get_class_accuracy(cls, k, predictions, targets):

        # extract relevant indices
        k_indices = [idx for idx, target in enumerate(targets) if np.argmax(target) == k]
        k_targets = np.asarray(targets)[k_indices]
        k_predictions = np.asarray(predictions)[k_indices]

        # calculate acc
        k_acc = cls.calculate_acc(k_predictions, k_targets)

        return k_acc

    # calculate accuracy based on prediction and target vector
    @classmethod
    def calculate_acc(cls, predictions, targets):
        boolean_array = [1 if (np.argmax(pred) == np.argmax(label)) else 0 for pred, label in zip(predictions, targets)]
        count_dict = Counter(boolean_array)
        try:
            acc = count_dict[1] / (count_dict[0] + count_dict[1])
        except Exception as e:
            print(e)
            acc = np.nan
        return acc

    # calculate softmax probabilities
    @classmethod
    def softmax(cls, x):
        e_x = np.exp(x)
        ex_sums = np.sum(e_x, axis=1)
        return e_x / ex_sums[:, np.newaxis]

    @classmethod
    def forward_pass(cls, W, batch):
        return np.dot(W, batch.T).T

    @classmethod
    def pass_through_network(cls, data_batch, label_batch, weight_matrix, weight_decay_factor):

        # pass data batch through network
        predictions = cls.softmax(cls.forward_pass(weight_matrix, data_batch))
        targets = label_batch

        # calculate data batch loss
        data_batch_loss = np.sum([np.dot(target, np.log(prediction))
                                  for target, prediction in zip(targets, predictions)])

        # calculate softmax cross entropy loss with l2 weight decay regularization
        W = weight_matrix
        l2_weight_decay_regularization = weight_decay_factor * 0.5 * np.sum(
            np.square(np.reshape(W, newshape=(W.shape[0] * W.shape[1],))))
        softmax_cross_entropy_loss = (-1) * data_batch_loss + l2_weight_decay_regularization

        # normalize loss so that different batch sizes are comparable
        softmax_cross_entropy_loss = softmax_cross_entropy_loss / data_batch.shape[0]

        return softmax_cross_entropy_loss, predictions, targets

    # shuffle data and labels
    @classmethod
    def shuffle_data(cls, data, labels):
        indices = list(range(len(data)))
        np.random.shuffle(indices)

        data_shuffled = data[indices]
        labels_shuffled = labels[indices]

        return data_shuffled, labels_shuffled

    # divide data and labels into mini batches
    @classmethod
    def create_mini_batches(cls, data, labels, mini_batch_size):
        data_mini_batches = []
        labels_mini_batches = []
        start_index = 0
        while start_index < len(data):
            data_mini_batches.append(data[start_index:start_index + mini_batch_size])
            labels_mini_batches.append(labels[start_index:start_index + mini_batch_size])
            start_index += mini_batch_size

        return np.asarray(data_mini_batches), np.asarray(labels_mini_batches)


if __name__ == '__main__':
    classifier = SoftmaxClassifier(model_name='Softmax_Classifier',
                                   num_classes=3,
                                   dimension_data_point=3,
                                   learning_rate=0.01,
                                   momentum_rate=0.5,
                                   mini_batch_size=20,
                                   epochs=1000,
                                   weight_decay_factor=0.0001)
    classifier.train()
    classifier.plot_train_history()
    classifier.plot_decision_boundaries()
