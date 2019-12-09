using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworkFromScratch.Models
{
    public class Network
    {
        public Network()
        {
            Layers = new List<Layer>();
        }
        public List<Layer> Layers { get; set; }
        public double TotalError { get; set; }

        public void UpdateWeights(double learningRate)
        {
            foreach (Layer layer in Layers)
            {
                foreach (Connection connection in layer.Connections)
                {
                    connection.Weight = connection.Weight - learningRate * connection.NewWeight;
                    connection.NewWeight = 0;
                }
            }
        }
    }
}
