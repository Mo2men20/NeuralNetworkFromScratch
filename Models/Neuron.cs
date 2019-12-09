using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworkFromScratch.Models
{
    public enum NeuronType
    {
        Normal=1,
        Bias=2
    }
    public class Neuron
    {
        public Neuron()
        {
            this.Id = Guid.NewGuid();
            this.Type = NeuronType.Normal;
        }

        public Guid Id { get; set; }
        public NeuronType Type { get; set; }
        public float Value { get; set; }
        public float DerivativeValue { get; set; }
        public double Error { get; set; }
        public float DesiredValue { get; set; }
        public float ErrorDerivativeValue { get; set; }
    }
}
