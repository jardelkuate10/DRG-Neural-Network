import random
import math
from sklearn.model_selection import train_test_split

class Neuron:
    def __init__(self, bias):
        self.bias = bias
        self.weights = []
        self.output = 0
        self.inputs = []

    def calculate_total_net_input(self, inputs):
        self.inputs = inputs
        total = 0

        for inpt in range(len(self.inputs)):
            total += self.inputs[inpt] * self.weights[inpt]

        return total + self.bias

    def squash(self, total_net_input):
        return 1 / (1 + math.exp(-total_net_input))

    def calculate_output(self, inputs):
        self.output = self.squash(self.calculate_total_net_input(inputs))

        return self.output

    def calculate_error(self, target_output):
        return 0.5 * (target_output - self.output) ** 2

    def calculate_pd_net_input_wrt_weight(self, index):
        return self.inputs[index]

    def calculate_pd_output_wrt_net_input(self):
        return self.output * (1 - self.output)

    def calculate_pd_error_wrt_output(self, target_output):
        return -(target_output - self.output)

    def calculate_delta(self, target_output):
        return self.calculate_pd_error_wrt_output(target_output) * self.calculate_pd_output_wrt_net_input()


class NeuronLayer:
    def __init__(self, num_neuron_per_layer, bias):
        self.bias = bias
        self.neurons = []

        for i in range(num_neuron_per_layer):
            self.neurons.append(Neuron(self.bias))

    def feed_forward(self, inputs):
        outputs = []
        for n in range(len(self.neurons)):
            outputs.append(self.neurons[n].calculate_output(inputs))
        return outputs


class NeuralNet:
    LR = 0.5

    def __init__(self, data, training_percent, max_iterations, num_hidden_layers, num_neuron_per_layer):
        self.training_data, self.testing_data = train_test_split(data, train_size=training_percent)
        self.training_inputs, self.training_outputs = self.create_training_sets()
        self.testing_inputs, self.testing_outputs = self.create_testing_sets()

        self.num_hidden_layers = num_hidden_layers
        self.num_neuron_per_layer = num_neuron_per_layer

        self.num_inputs = len(self.testing_inputs[0])
        self.max_iterations = max_iterations

        self.hidden_layers = self.create_hidden_layers()
        self.output_layer = NeuronLayer(1, 1)

    def create_training_sets(self):
        training_sets = []
        training_inputs = []
        training_outputs = []
        for index, row in self.training_data.iterrows():
            training_sets.append(
                ((row['sepallength'], row['sepalwidth'], row['petallength'], row['petalwidth']),
                 row['class']))  # this will change for different data sets

        for tr_set in range(len(training_sets)):
            training_inputs.append(training_sets[tr_set][0])
            training_outputs.append(training_sets[tr_set][1])

        return training_inputs, training_outputs

    def create_testing_sets(self):
        testing_sets = []
        testing_inputs = []
        testing_outputs = []
        for index, row in self.testing_data.iterrows():
            testing_sets.append(
                ((row['sepallength'], row['sepalwidth'], row['petallength'], row['petalwidth']),
                 row['class']))  # this will change for different data sets

        for te_set in range(len(testing_sets)):
            testing_inputs.append(testing_sets[te_set][0])
            testing_outputs.append(testing_sets[te_set][1])

        return testing_inputs, testing_outputs

    def create_hidden_layers(self):
        hidden_layers = []
        for i in range(self.num_hidden_layers):
            hidden_layers.append(NeuronLayer(self.num_neuron_per_layer, 1))

        return hidden_layers

    def init_weights_from_inputs_to_hidden_layer_neurons(self):
        for h in range(len(self.hidden_layers[0].neurons)):
            for i in range(self.num_inputs):
                self.hidden_layers[0].neurons[h].weights.append(random.random())

    def init_weights_from_hidden_layer_neurons_to_hidden_layer_neurons(self):
        for h in range(1, self.num_hidden_layers):
            for i in range(len(self.hidden_layers[h].neurons)):
                for j in range(len(self.hidden_layers[h - 1].neurons)):
                    self.hidden_layers[h].neurons[i].weights.append(random.random())

    def init_weights_from_hidden_layer_neurons_to_output_layer_neurons(self):
        for o in range(len(self.output_layer.neurons)):
            for h in range(len(self.hidden_layers[-1].neurons)):
                self.output_layer.neurons[o].weights.append(random.random())

    def feed_forward(self, inputs):
        output0 = inputs
        output1 = self.hidden_layers[0].feed_forward(output0)

        for h in range(1, self.num_hidden_layers, 2):
            try:
                output0 = self.hidden_layers[h].feed_forward(output1)

                output1 = self.hidden_layers[h + 1].feed_forward(output0)

                if (h + 1) not in range(self.num_hidden_layers):
                    return self.output_layer.feed_forward(output0)

            finally:
                return self.output_layer.feed_forward(output1)

    def train(self):
        for tr_in in range(len(self.training_inputs)):
            self.feed_forward(self.training_inputs[tr_in])

            # output layer neuron delta
            delta_output = [0] * len(self.output_layer.neurons)
            for o in range(len(self.output_layer.neurons)):
                delta_output[o] = self.output_layer.neurons[o].calculate_delta(self.training_outputs[tr_in])

            hidden_deltas = []
            hidden_dl = [0] * len(self.hidden_layers[0].neurons)

            # last hidden layer neuron delta
            for n in range(len(self.hidden_layers[-1].neurons)):
                index = 0
                for o in range(len(self.output_layer.neurons)):
                    index += delta_output[o] * self.output_layer.neurons[o].weights[n]
                hidden_dl[n] = index * self.hidden_layers[-1].neurons[n].calculate_pd_output_wrt_net_input()
            hidden_deltas.append(hidden_dl)

            # middle hidden layers neuron delta
            for i in reversed(range(len(self.hidden_layers) - 1)):
                hidden_delta = [0] * len(self.hidden_layers[0].neurons)
                for n in range(len(self.hidden_layers[i].neurons)):
                    index = 0
                    for o in range(len(self.hidden_layers[i + 1].neurons)):
                        index += hidden_deltas[0][o] * self.hidden_layers[i + 1].neurons[o].weights[n]
                    hidden_delta[n] = index * self.hidden_layers[i].neurons[n].calculate_pd_output_wrt_net_input()
                hidden_deltas.insert(0, hidden_delta)

            # update output neuron weights
            for o in range(len(self.output_layer.neurons)):
                for w_ho in range(len(self.output_layer.neurons[o].weights)):
                    pd_error_wrt_weight = delta_output[o] * self.output_layer.neurons[
                        o].calculate_pd_net_input_wrt_weight(w_ho)

                    self.output_layer.neurons[o].weights[w_ho] -= self.LR * pd_error_wrt_weight

            # update hidden neuron weights
            for h in range(len(self.hidden_layers)):
                for n in range(len(self.hidden_layers[h].neurons)):
                    for w in range(len(self.hidden_layers[h].neurons[n].weights)):
                        pd_error_wrt_weight = hidden_deltas[h][n] * self.hidden_layers[h].neurons[
                            n].calculate_pd_net_input_wrt_weight(w)

                        self.hidden_layers[h].neurons[n].weights[w] -= self.LR * pd_error_wrt_weight

    def calculate_total_error(self):
        total_error = 0
        for tr in range(len(self.training_inputs)):
            self.feed_forward(self.training_inputs[tr])
            for n in range(len(self.output_layer.neurons)):
                total_error += self.output_layer.neurons[n].calculate_error(self.training_outputs[tr])

        return total_error

    def test(self):
        total_error = 0
        for te in range(len(self.testing_inputs)):
            self.feed_forward(self.testing_inputs[te])
            for n in range(len(self.output_layer.neurons)):
                total_error += self.output_layer.neurons[n].calculate_error(self.testing_outputs[te])

        return total_error

    def report(self):
        t = self.test()
        layer = ''
        for h in range(len(self.hidden_layers)):
            if h == 0:
                layer = "1st Hidden"
            elif h == len(self.hidden_layers):
                layer = "Last Hidden"
            elif h == 1:
                layer = "2nd Hidden"
            elif h == 2:
                layer = "3rd Hidden"
            else:
                layer = f"{h + 1}th Hidden"

            print(
                f'''
Layer {h + 1} ({layer} Layer):
                        ''')

            for n in range(len(self.hidden_layers[h].neurons)):
                print(
                    f'''
    Neuron {n + 1} weights: {self.hidden_layers[h].neurons[n].weights}
                        ''')

        print(
            f'''
Layer {h + 2} (Output Layer)
                        ''')
        for i in range(len(self.output_layer.neurons)):
            print(
                f'''
    Neuron {i + 1} weights: {self.output_layer.neurons[i].weights}
                    ''')

        print(f"Total training error = {self.calculate_total_error()}")

        print(f"Total test error Iris = {t}")

        return t

########################################################################################################################

# EXAMPLE RUN

#nn = NeuralNet(df, 0.8, 1000, 3, 5)

#nn.init_weights_from_inputs_to_hidden_layer_neurons()
#nn.init_weights_from_hidden_layer_neurons_to_hidden_layer_neurons()
#nn.init_weights_from_hidden_layer_neurons_to_output_layer_neurons()

#test_errors = []

#for i in range(nn.max_iterations):
#    nn.train()
#nn.report()