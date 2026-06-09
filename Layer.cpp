#include "Perceptron.h"

class Layer
{
    private:
        std::vector<Perceptron> m_perceptrons;
    public:
        // layer constructor
        Layer(int numInputs, int numOutputs)
        {
            m_perceptrons.reserve(numOutputs);
            for (int i = 0; i < numOutputs; ++i)
            {
                m_perceptrons.emplace_back(numInputs);
            }
        }

        /*
        void printNeurons()
        {
            for (auto i = m_perceptrons.begin(); i != m_perceptrons.end(); ++i)
            {
                (*i).printParams();
            }
        }
        */

        std::vector<std::shared_ptr<Value>> parameters() const
        {
            std::vector<std::shared_ptr<Value>> params;
            for (const auto& perceptron : m_perceptrons)
            {
                auto p_params = perceptron.parameters();
                params.insert(params.end(), p_params.begin(), p_params.end());
            }
            return params;
        }

        // return a vector of outputs for all perceptrons
        std::vector<std::shared_ptr<Value>> predictNext(const std::vector<std::shared_ptr<Value>>& in) const
        {
            std::vector<std::shared_ptr<Value>> out;
            for (auto i = m_perceptrons.begin(); i != m_perceptrons.end(); ++i)
            {
                out.push_back((*i).predictNext(in));
            }
            return out;
        }
};

int main()
{
    Layer layer(4, 2);
    // layer.printNeurons();
    auto a = std::make_shared<Value>(1.0);
    auto b = std::make_shared<Value>(1.0);
    auto c = std::make_shared<Value>(1.0);
    auto d = std::make_shared<Value>(1.0);
    std::vector<std::shared_ptr<Value>> vector = { a, b, c, d };
    auto out = layer.predictNext(vector);
    for (auto i = out.begin(); i != out.end(); ++i)
    {
        std::cout << (*i) << " ";
    }
    std::cout << std::endl;
}