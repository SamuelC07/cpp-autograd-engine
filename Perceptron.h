#pragma once
#include "Value.h"
#include <iostream>
#include <vector>
#include <memory>
#include <random>

class Perceptron
{
    private:
        std::vector<std::shared_ptr<Value>> m_weights;
        std::shared_ptr<Value> m_b;
    public:
        Perceptron(int numWeights)
        {
            std::uniform_real_distribution<double> unif(-1, 1);
            std::random_device rd;
            std::default_random_engine re(rd());
            for (int i = 0; i < numWeights; ++i)
            {
                m_weights.emplace_back(std::make_shared<Value>(unif(re)));
            }
            m_b = std::make_shared<Value>(unif(re));
        }

        std::shared_ptr<Value> predictNext(const std::vector<std::shared_ptr<Value>>& inputs) const
        {
            if (m_weights.size() != inputs.size())
            {
                throw std::out_of_range("size of values must be equal to number of perceptron weights");
            }
            std::shared_ptr<Value> sum = m_b;
            for (int i = 0; i < inputs.size(); ++i)
            {
                sum = sum + (m_weights.at(i) * inputs.at(i));
            }
            return sum->LeakyReLU();
        }
    
        std::vector<std::shared_ptr<Value>> parameters() const
        {
            std::vector<std::shared_ptr<Value>> params;
            params.reserve(1 + m_weights.size());
            params.push_back(m_b);
            for (const auto& i : m_weights)
            {
                params.push_back(i);
            }
            return params;
        }

        /*
        void printB()
        {
            std::cout << "b is " << m_b << std::endl;
        }

        // print weights
        void printWeights()
        {
            std::cout << "weights are: ";
            for (auto i = m_weights.begin(); i != m_weights.end(); ++i)
            {
                std::cout << (*i) << " ";
            }
            std::cout << std::endl;
        }

        void printParams()
        {
            printWeights();
            printB();
        }
        */
};