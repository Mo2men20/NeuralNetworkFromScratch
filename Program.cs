using NeuralNetworkFromScratch.Models;
using System;
using System.Linq;

namespace NeuralNetworkFromScratch
{
    class Program
    {
        static void Main(string[] args)
        {
            Network network = new Network();

            Layer inputLayer = new Layer();
            inputLayer.Type = LayerType.Input;

            Neuron input1 = new Neuron();
            input1.Value = 0.05f ;
            Neuron input2 = new Neuron();
            input2.Value = 0.1f;
            Neuron inputBias = new Neuron();
            inputBias.Value = 1;
            inputBias.Type = NeuronType.Bias;

            inputLayer.Neurons.Add(input1);
            inputLayer.Neurons.Add(input2);
            inputLayer.Neurons.Add(inputBias);

            Layer hiddenLayer = new Layer();
            hiddenLayer.PreviousLayer = inputLayer;
            hiddenLayer.Type = LayerType.Hidden;

            Neuron hidden1 = new Neuron();
            Neuron hidden2 = new Neuron();
            Neuron hiddenBias = new Neuron();
            hiddenBias.Value = 1;
            hiddenBias.Type = NeuronType.Bias;

            hiddenLayer.Neurons.Add(hidden1);
            hiddenLayer.Neurons.Add(hidden2);
            hiddenLayer.Neurons.Add(hiddenBias);

            Layer outputLayer = new Layer();
            outputLayer.PreviousLayer = hiddenLayer;
            outputLayer.Type = LayerType.Output;

            Neuron output1 = new Neuron();
            Neuron output2 = new Neuron();
            output1.DesiredValue = 0.01f;
            output2.DesiredValue = 0.99f;
            outputLayer.Neurons.Add(output1);
            outputLayer.Neurons.Add(output2);

            network.Layers.Add(inputLayer);
            network.Layers.Add(hiddenLayer);
            network.Layers.Add(outputLayer);


            inputLayer.Connections.Add(new Connection { From = input1, To = hidden1, Weight= 0.15, Name = "W1" });
            inputLayer.Connections.Add(new Connection { From = input1, To = hidden2, Weight = 0.25, Name = "W3" });
            inputLayer.Connections.Add(new Connection { From = input2, To = hidden1, Weight = 0.20, Name = "W2" });
            inputLayer.Connections.Add(new Connection { From = input2, To = hidden2, Weight = 0.30, Name = "W4" });
            inputLayer.Connections.Add(new Connection { From = inputBias, To = hidden1,Weight = .35 });
            inputLayer.Connections.Add(new Connection { From = inputBias, To = hidden2, Weight = .35 });

            hiddenLayer.Connections.Add(new Connection { From = hidden1, To = output1, Weight =0.40,Name="W5" });
            hiddenLayer.Connections.Add(new Connection { From = hidden2, To = output1, Weight = 0.45, Name = "W6" });
            hiddenLayer.Connections.Add(new Connection { From = hiddenBias, To = output1, Weight = .6 });
            hiddenLayer.Connections.Add(new Connection { From = hidden1, To = output2, Weight = 0.5, Name = "W7" });
            hiddenLayer.Connections.Add(new Connection { From = hidden2, To = output2, Weight =  0.55, Name = "W8" });
            hiddenLayer.Connections.Add(new Connection { From = hiddenBias, To = output2, Weight = .6 });


            Random random = new Random();

            //foreach (Layer layer in network.Layers)
            //{
            //    foreach (Connection conn in layer.Connections)
            //    {
            //        conn.Weight = random.NextDouble();
            //    }
            //}

            //Pulse(network);
            //CalculateError(network);
            //CalculateNewWeights(network);
            //network.UpdateWeights(.5);

            for (int i = 0; i < 10000; i++)
            {
                Pulse(network);
                CalculateError(network);
                CalculateNewWeights(network);
                network.UpdateWeights(.5);
            }

        }


        static void Pulse(Network network)
        {
            foreach (Layer layer in network.Layers)
            {
                if (layer.Type == LayerType.Input) continue;

                foreach (Neuron neuron in layer.Neurons)
                {
                    if (neuron.Type == NeuronType.Bias) continue;

                    var nConnections = layer.PreviousLayer.Connections.Where(c => c.To.Id.Equals(neuron.Id)).ToList();

                    neuron.Value = Sigmoid(nConnections.Sum(n => n.From.Value * n.Weight));
                }
            }
        }

        static void CalculateError(Network network)
        {
            foreach (Layer layer in network.Layers)
            {
                if (layer.Type == LayerType.Input || layer.Type == LayerType.Hidden) continue;

                double totalError = 0;
                foreach (Neuron neuron in layer.Neurons)
                {
                    if (neuron.Type == NeuronType.Bias) continue;

                    neuron.Error = Math.Pow((neuron.DesiredValue - neuron.Value), 2) / 2;
                    totalError += neuron.Error;
                    neuron.ErrorDerivativeValue = ErrorDerivative(neuron.DesiredValue, neuron.Value);
                    neuron.DerivativeValue = SigmoidDerivative(neuron.Value);
                }

                network.TotalError = (double)totalError;
            }
        }

        static void CalculateNewWeights(Network network)
        {
            foreach (Layer layer in network.Layers)
            {
                if (layer.Type == LayerType.Input) continue;

                switch (layer.Type)
                {
                    case LayerType.Output:
                        foreach (Neuron neuron in layer.Neurons)
                        {                         
                            var connections = layer.PreviousLayer.Connections.Where(c => c.From.Type != NeuronType.Bias && c.To.Id.Equals(neuron.Id)).ToList();

                            foreach (Connection connection in connections)
                            { 
                                connection.NewWeight = ErrorDerivative(neuron.DesiredValue, neuron.Value) * SigmoidDerivative(neuron.Value) *  connection.From.Value;
                            }
                        }
                    break;
                    case LayerType.Hidden:
                        foreach (Neuron neuron in layer.Neurons)
                        {
                            var connections = layer.Connections.Where(c => c.From.Type != NeuronType.Bias && c.From.Id.Equals(neuron.Id)).ToList();

                            double totalErrorDerivative = 0;

                            foreach (Connection connection in connections)
                            {
                                totalErrorDerivative += connection.To.ErrorDerivativeValue * connection.To.DerivativeValue * connection.Weight;
                            }

                            var previousConnections = layer.PreviousLayer.Connections.Where(c => c.From.Type != NeuronType.Bias && c.To.Id.Equals(neuron.Id)).ToList();
                            foreach (Connection connection in previousConnections)
                            {
                                
                                connection.NewWeight = totalErrorDerivative * SigmoidDerivative(neuron.Value) * connection.From.Value;
                            }

                        }
                        break;
                }
            }
        }


        public static float Sigmoid(double value)
        {
            float k = (float)Math.Exp(value);
            return k / (1.0f + k);
        }

        public static float SigmoidDerivative(float value)
        {
            return value * (1f - value);
        }

        public static float ErrorDerivative(float target, float value)
        {
            return -1 * (target - value);
        }


    }
}
