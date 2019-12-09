using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworkFromScratch.Models
{
    public enum LayerType
    {
        Input=1,
        Hidden=2,
        Output=3
    }
    public class Layer
    {
        public Layer()
        {
            Neurons = new List<Neuron>();
            Connections = new List<Connection>();
        }
        public Layer PreviousLayer { get; set; }
        public List<Neuron> Neurons { get; set; }
        public LayerType Type { get; set; }
        public List<Connection> Connections { get; set; }
    }
}
