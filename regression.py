import os
import sys
import tqdm
import numpy as np
import matplotlib.pyplot as plt

from operator import itemgetter
from collections import Counter


# function to calculate mse
def music_mse(prediction, gt):
    prediction_rounded = np.rint(prediction)
    squared_differences = np.square(prediction_rounded - gt)
    return np.mean(squared_differences)


# function to plot histograms to compare trained weights
def weight_vector_histograms(l1_weights, l2_weights):

    # create figure
    f, (ax1, ax2) = plt.subplots(1, 2)

    # plot figure
    ax1.hist(l1_weights, bins=25, rwidth=0.95, alpha=0.1, color='blue', label='l1-regularized weights')
    ax1.set_xlabel('Bins')
    ax1.set_ylabel('#Weights in bin')
    ax1.set_title('Weight Histogram for L1-Regularized Regression')

    ax2.hist(l2_weights, bins=25, rwidth=0.95, alpha=0.1, color='red', label='l2-regularized weights')
    ax2.set_xlabel('Bins')
    ax2.set_ylabel('#Weights in bin')
    ax2.set_title('Weight Histogram for L2-Regularized Regression')

    # show figure
    plt.show()


class DataLoader:

    def __init__(self, f_name, add_bias):
        self.f_name = f_name
        self.add_bias = add_bias
        self.complete_dataset = None
        self.train_years = None
        self.train_features = None
        self.test_years = None
        self.test_features = None
        self.represented_years = None

    def load_music_data(self):

        # load data set from txt-file into a numpy array
        self.complete_dataset = np.loadtxt(fname=os.path.join('./datasets', self.f_name), delimiter=',')

        # add bias if wanted
        if self.add_bias:
            bias_matrix = np.ones(shape=(self.complete_dataset.shape[0], 1))
            self.complete_dataset = np.concatenate((self.complete_dataset, bias_matrix), axis=1)

        # split data set into train and test set
        train_set, test_set = self.complete_dataset[:463714], self.complete_dataset[463714:]

        # split train and test set into features and labels
        self.train_years = train_set[:, 0].astype(dtype=np.int)
        self.train_features = train_set[:, 1:].astype(dtype=np.float)
        self.test_years = test_set[:, 0].astype(dtype=np.int)
        self.test_features = test_set[:, 1:].astype(dtype=np.float)

        return self.train_features, self.train_years, self.test_features, self.test_years

    def evaluate_dataset(self):

        # print variable range
        print('Minimum feature value in dataset:', np.min(self.complete_dataset[:, 1:]))
        print('Maximum feature value in dataset:', np.max(self.complete_dataset[:, 1:]))

        # get represented years in data set
        self.represented_years = []
        for year in self.complete_dataset[:, 0]:
            if year not in self.represented_years:
                self.represented_years.append(year)
        self.represented_years = np.rint(np.sort(self.represented_years))
        print('Sorted years that are present in dataset:', self.represented_years)

        # plot year label histogram
        all_year_labels = self.complete_dataset[:, 0]
        plt.hist(all_year_labels, bins=15, rwidth=0.95, color='blue')
        plt.xlabel('Bins')
        plt.ylabel('#Years in bin')
        plt.title('Label Histogram')
        plt.show()

        # mean squared error output if classifier always outputs most common year
        year_count_dict = Counter(self.complete_dataset[:, 0])
        most_common_year = sorted(year_count_dict.items(), key=itemgetter(1), reverse=True)[0][0]
        predictions = np.full(shape=(self.test_years.shape[0],), fill_value=np.rint(most_common_year))
        targets = np.rint(self.test_years)
        print('MSE if classifier always outputs most common year:', music_mse(predictions, targets))

        # mean squared error out if classifier always output 1998
        predictions = np.full(shape=(self.test_years.shape[0],), fill_value=1998)
        targets = np.rint(self.test_years)
        print('MSE if classifier always outputs 1998:', music_mse(predictions, targets))


class Regression:

    # load data
    X_train, y_train, X_test, y_test = DataLoader(f_name='YearPredictionMSD.txt',
                                                  add_bias=True).load_music_data()

    def __init__(self,
                 model_name,
                 regression_type,
                 input_data_size,
                 learning_rate,
                 momentum_rate,
                 mini_batch_size,
                 epochs,
                 weight_decay_factor,
                 use_sgd):
        self.model_name = model_name
        self.regression_type = regression_type
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.previous_gradient = np.zeros(shape=(input_data_size,))
        self.mini_batch_size = mini_batch_size
        self.epochs = epochs
        self.weight_decay_factor = weight_decay_factor
        self.weight_vector = np.random.uniform(low=-0.01, high=0.01, size=(input_data_size, ))
        self.use_sgd = use_sgd
        self.train_loss_history = None
        self.test_loss_history = None

        # standardize data
        x_train_mean = np.mean(self.X_train, axis=0)
        x_train_std = np.std(self.X_train, axis=0)
        self.X_train = (self.X_train - x_train_mean) / x_train_std
        self.X_test = (self.X_test - x_train_mean) / x_train_std

        # make bias one again (will be zero after standardization)
        self.X_train[:, -1] = np.ones(shape=(self.X_train.shape[0],))
        self.X_test[:, -1] = np.ones(shape=(self.X_test.shape[0],))

        # subtract 1922 from labels so that gradient doesn't explode
        if self.regression_type == 'poisson':
            self.y_train -= 1922
            self.y_test -= 1922

    def train(self):

        # calculate closed form optimal solution
        if not self.use_sgd:

            # ======= CLOSED FORM SOLUTION START =======

            if self.regression_type == 'ridge':
                self.weight_vector = np.dot(np.dot(np.linalg.inv(np.dot(self.X_train.T,
                                                                        self.X_train) +
                                                                 self.weight_decay_factor *
                                                                 np.identity(self.X_train.shape[1])),
                                                   self.X_train.T),
                                            self.y_train)
            elif self.regression_type == 'lasso':
                self.weight_vector = np.dot(np.linalg.inv(np.dot(self.X_train.T,
                                                                 self.X_train)),
                                            (np.dot(self.X_train.T, self.y_train) - (self.weight_decay_factor / 2)))
            elif self.regression_type == 'poisson':
                print('No closed form solution for poisson regression.')
                return
            else:
                print('Correct regression types are \'ridge\' or \'lasso\' and \'poisson\'.')
                return

            # get train loss
            train_loss, train_mse = self.calculate_loss(self.X_train, self.y_train,
                                                        self.weight_vector, self.regression_type,
                                                        self.weight_decay_factor)
            print('Optimal Train Loss:', train_loss)
            print('Optimal Train MSE:', train_mse)

            # get test loss
            test_loss, test_mse = self.calculate_loss(self.X_test, self.y_test,
                                                      self.weight_vector, self.regression_type,
                                                      self.weight_decay_factor)
            print('Optimal Test Loss:', test_loss)
            print('Optimal Test MSE:', test_mse)

            print('Trained weights:', self.weight_vector)

            # stop training
            return

            # ======= CLOSED FORM SOLUTION END =======

        # ======= STOCHASTIC GRADIENT DESCENT START =======

        # train and test data history
        self.train_loss_history = []
        self.test_loss_history = []

        for epoch in range(self.epochs):

            # ======= EPOCH START =======

            if self.learning_rate % 50 == 0:
                self.learning_rate = self.learning_rate / 2

            print('Epoch:', epoch)

            # prepare training data for epoch
            data_shuffled, labels_shuffled = self.shuffle_data(self.X_train, self.y_train)
            data_mini_batches, labels_mini_batches = self.create_mini_batches(data_shuffled, labels_shuffled,
                                                                              self.mini_batch_size)

            mini_batch_loss_list = []
            mini_batch_mse_list = []
            for data_batch, label_batch in tqdm.tqdm(zip(data_mini_batches, labels_mini_batches)):

                # ======= MINI BATCH START =======

                # calculate loss for mini batch
                loss, mse = self.calculate_loss(data_batch, label_batch,
                                                self.weight_vector, self.regression_type, self.weight_decay_factor)
                mini_batch_loss_list.append(loss)
                mini_batch_mse_list.append(mse)

                # calculate gradient for weight update
                if self.regression_type == 'ridge':
                    gradient = 2 * (np.dot((np.dot(data_batch.T, data_batch) +
                                            self.weight_decay_factor * np.identity(data_batch.shape[1])),
                                           self.weight_vector) -
                                    np.dot(data_batch.T, label_batch))
                elif self.regression_type == 'lasso':
                    gradient = 2 * (np.dot(np.dot(data_batch.T, data_batch),
                                           self.weight_vector) - np.dot(data_batch.T, label_batch)) + \
                               self.weight_decay_factor
                elif self.regression_type == 'poisson':
                    gradient = np.dot(data_batch.T, np.exp(np.dot(data_batch, self.weight_vector))) - \
                                                   np.dot(label_batch, data_batch)
                    gradient = np.clip(gradient, -0.5, 0.5)
                else:
                    print('Correct regression types are \'ridge\' or \'lasso\' and \'poisson\'.')
                    sys.exit(0)
                momentum_corrected_gradient = (self.momentum_rate * self.previous_gradient) + \
                                              (1 - self.momentum_rate) * gradient
                self.previous_gradient = gradient
                self.weight_vector = self.weight_vector - self.learning_rate * momentum_corrected_gradient

                # ======= MINI BATCH END =======

            print('Average Train Loss:', np.mean(mini_batch_loss_list))
            print('Average Train MSE:', np.mean(mini_batch_mse_list))
            self.train_loss_history.append(np.mean(mini_batch_loss_list))

            # ======= TEST SET EVALUATION START =======

            # test set evaluation after epoch
            test_loss, test_mse = self.calculate_loss(self.X_test, self.y_test,
                                                      self.weight_vector, self.regression_type,
                                                      self.weight_decay_factor)
            print('Test Loss:', test_loss)
            print('Test MSE:', test_mse)
            self.test_loss_history.append(test_loss)

            # ======= TEST SET EVALUATION END

            # ======= EPOCH END =======

        # ======= STOCHASTIC GRADIENT DESCENT END =======

    def plot_train_history(self):

        # plot left figure with loss
        plt.plot(list(range(0, self.epochs, 1)), self.train_loss_history)
        plt.xlabel('Epochs')
        plt.title('Average Train Loss')

        # plot figure
        plt.show()

    # plot histogram for correct test years and wrong test years
    def plot_right_wrong_prediction_distribution(self):

        # calculate test predictions
        if self.regression_type == 'poisson':
            test_predictions = np.rint(np.exp(np.dot(self.X_test, self.weight_vector))) + 1922
            test_targets = self.y_test + 1922
        else:
            test_predictions = np.rint(np.dot(self.X_test, self.weight_vector))
            test_targets = self.y_test

        # extract right and wrongly prediction years
        correct_years_list = []
        wrong_years_list = []
        for prediction, target in zip(test_predictions, test_targets):
            if prediction == target:
                correct_years_list.append(target)
            else:
                wrong_years_list.append(target)

        # create figure
        f, (ax1, ax2) = plt.subplots(1, 2)

        # plot figure
        ax1.hist(correct_years_list, bins=25, rwidth=0.95, alpha=0.1, color='green', label='correctly predicted years')
        ax1.set_xlabel('Bins')
        ax1.set_ylabel('#Years in bin')
        ax1.set_title('Histogram for ' + str(len(correct_years_list)) + ' Correctly Predicted Years')

        ax2.hist(wrong_years_list, bins=25, rwidth=0.95, alpha=0.1, color='red', label='wrongly predicted years')
        ax2.set_xlabel('Bins')
        ax2.set_ylabel('#Years in bin')
        ax2.set_title('Histogram for ' + str(len(wrong_years_list)) + ' Wrongly Predicted Years')

        plt.suptitle(self.model_name)

        # show figure
        plt.show()

    # calculate predictions with current weights and calculate loss
    @classmethod
    def calculate_loss(cls, data, labels, weight_vector, regression_type, weight_decay_factor):

        # calculate loss
        if regression_type == 'ridge':
            predictions = np.rint(np.dot(data, weight_vector))
            loss = (np.sum(np.square(predictions - labels)) +
                    weight_decay_factor * np.sum(np.square(weight_vector))) / (data.shape[0])
        elif regression_type == 'lasso':
            predictions = np.rint(np.dot(data, weight_vector))
            loss = (np.sum(np.square(predictions - labels)) +
                    weight_decay_factor * np.sum(np.abs(weight_vector))) / (data.shape[0])
        elif regression_type == 'poisson':
            predictions = np.rint(np.exp(np.dot(data, weight_vector)))
            loss = (np.sum(np.exp(np.dot(data, weight_vector)) - np.multiply(labels, predictions))) / (data.shape[0])
        else:
            print('Correct regression types are \'ridge\' or \'lasso\' and \'poisson\'.')
            sys.exit(0)

        # calculate mse
        mse = music_mse(predictions, labels)

        return loss, mse

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

    regression_model_rr = Regression(model_name='Ridge Regression',
                                     regression_type='ridge',
                                     input_data_size=91,
                                     learning_rate=0.000001,
                                     momentum_rate=0.1,
                                     mini_batch_size=8,
                                     epochs=5,
                                     weight_decay_factor=0.0001,
                                     use_sgd=True)
    regression_model_rr.train()
    regression_model_rr.plot_right_wrong_prediction_distribution()
    regression_model_rr.plot_train_history()

    regression_model_lr = Regression(model_name='Lasso Regression',
                                     regression_type='lasso',
                                     input_data_size=91,
                                     learning_rate=0.000001,
                                     momentum_rate=0.1,
                                     mini_batch_size=8,
                                     epochs=5,
                                     weight_decay_factor=0.0001,
                                     use_sgd=True)
    regression_model_lr.train()
    regression_model_lr.plot_right_wrong_prediction_distribution()
    regression_model_lr.plot_train_history()

    regression_model_pr = Regression(model_name='Poisson Regression',
                                     regression_type='poisson',
                                     input_data_size=91,
                                     learning_rate=0.0001,
                                     momentum_rate=0.1,
                                     mini_batch_size=128,
                                     epochs=5,
                                     weight_decay_factor=0.0001,
                                     use_sgd=True)
    regression_model_pr.train()
    regression_model_pr.plot_right_wrong_prediction_distribution()
    regression_model_pr.plot_train_history()

    dl = DataLoader(f_name='YearPredictionMSD.txt', add_bias=True)
    dl.load_music_data()
    dl.evaluate_dataset()
