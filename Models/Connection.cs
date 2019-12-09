using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworkFromScratch.Models
{
    public class Connection
    {
        public Neuron From { get; set; }
        public Neuron To { get; set; }
        public double Weight { get; set; }
        public double NewWeight { get; set; }
        public string Name { get; set; }
    }
}
