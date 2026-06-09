#pragma once

#include "Layer.h"

class MLP
{
    private:
        std::vector<Layer> layers;
    public:
        MLP(int numInputs, std::vector<int> numOutputs)
        {
            layers.reserve(numOutputs.size());
            layers.emplace_back(numInputs, numOutputs.at(0));
            for (int i = 0; i < numOutputs.size() - 1; ++i)
            {
                layers.emplace_back(numOutputs.at(i), numOutputs.at(i + 1));
            }
        }

        /*
        void printLayers()
        {
            for (auto i = layers.begin(); i != layers.end(); ++i)
            {
               (*i).printNeurons();
            }
        }
        */

        std::vector<std::shared_ptr<Value>> parameters() const
        {
            std::vector<std::shared_ptr<Value>> params;
            for (const auto& layer : layers)
            {
                auto l_params = layer.parameters();
                params.insert(params.end(), l_params.begin(), l_params.end());
            }
            return params;
        }

        // return a vector of outputs for all perceptrons
        std::vector<std::shared_ptr<Value>> predict(const std::vector<std::shared_ptr<Value>>& in) const
        {
            auto x = in;
            for (int i = 0; i < layers.size(); ++i)
            {
                x = layers.at(i).predictNext(x);
            }
            return x;
        }
};