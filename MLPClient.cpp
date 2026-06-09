#include "MLP.h"

int main()
{
    std::vector<int> vec = { 8, 4, 1 };
    int numInputs = 2;
    MLP mlp(numInputs, vec);

    // inputs
    std::vector<double> xs = {
        // Quadrant 1 (Pos, Pos) -> Target: 1.0
        1.5, 2.3,    0.8, 1.9,    2.7, 0.5,    1.2, 1.2,    2.1, 1.7,    67, 67,
        // Quadrant 3 (Neg, Neg) -> Target: 1.0
        -1.2, -1.8,  -2.5, -0.9,  -0.5, -2.2,  -1.9, -1.9,  -2.2, -0.6,
        // Quadrant 2 (Neg, Pos) -> Target: 0.0
        -1.6, 2.1,   -0.7, 1.4,   -2.4, 0.8,   -1.1, 1.9,   -1.8, 1.1,
        // Quadrant 4 (Pos, Neg) -> Target: 0.0
        2.2, -1.5,   1.1, -2.4,   0.6, -0.9,   1.9, -1.2,   2.5, -2.1
    };

    // expected outputs
    std::vector<double> ys = {
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  // Q1 points
        1.0, 1.0, 1.0, 1.0, 1.0,  // Q3 points
        0.0, 0.0, 0.0, 0.0, 0.0,  // Q2 points
        0.0, 0.0, 0.0, 0.0, 0.0   // Q4 points
    };

    std::vector<std::shared_ptr<Value>> ypred(ys.size());
    std::vector<std::shared_ptr<Value>> inputs(numInputs);

    for (int k = 0; k < 2000; ++k)
    {
        // forward prop
        for (int r = 0; r < ys.size(); ++r)
        {
            for (int c = 0; c < numInputs; ++c)
            {
                inputs[c] = std::make_shared<Value>(xs[r * numInputs + c]);
            }
            ypred[r] = mlp.predict(inputs)[0];
        }

        // calculate loss
        auto loss = std::make_shared<Value>(0.0);
        for (int i = 0; i < ys.size(); ++i)
        {
            loss = loss + pow(ypred[i] - std::make_shared<Value>(ys[i]), 2);
        }

        // average loss out for large inputs
        auto final_loss = loss * std::make_shared<Value>(1.0 / ys.size());

        auto params = mlp.parameters();
        for (auto& p : params)
        {
            p->reset();
        }
        final_loss->backwards();

        // std::cout << "first weight grad: " << params[0]->getGradient() << std::endl;
        // loss->backwards();
        // std::cout << "Loss Grad: " << loss->getGradient() << " | Prediction Grad: " << ypred[0]->getGradient() << std::endl;

        for (auto& p : params)
        {
            p->addNum(-0.05 * p->getGradient());
        }
        std::cout << k << " " << final_loss << std::endl;
    }

    // auto pointA = std::make_shared<Value>(2);
    // auto pointB = std::make_shared<Value>(3);

    // auto pointC = std::make_shared<Value>(100);
    // auto pointD = std::make_shared<Value>(20);

    // auto pointE = std::make_shared<Value>(-2);
    // auto pointF = std::make_shared<Value>(-3);

    // auto pointG = std::make_shared<Value>(-1.2);
    // auto pointH = std::make_shared<Value>(-4.3);

    // std::vector<std::shared_ptr<Value>> ve1 = { pointA, pointB };
    // std::vector<std::shared_ptr<Value>> ve2 = { pointC, pointD };
    // std::vector<std::shared_ptr<Value>> ve3 = { pointE, pointF };
    // std::vector<std::shared_ptr<Value>> ve4 = { pointG, pointH };

    // std::cout << "point 1 is in first or third quadrant: " << mlp.predict(ve1)[0] << std::endl;
    // std::cout << "point 2 is in first or third quadrant: " << mlp.predict(ve2)[0] << std::endl;
    // std::cout << "point 3 is in first or third quadrant: " << mlp.predict(ve3)[0] << std::endl;
    // std::cout << "point 4 is in first or third quadrant: " << mlp.predict(ve4)[0] << std::endl;


    // --- INTERACTIVE EVALUATION LOOP ---
    std::cout << "\n==================================================" << std::endl;
    std::cout << "TRAINING COMPLETE! Entering Interactive Mode." << std::endl;
    std::cout << "Type two numbers (X Y) to test a custom point." << std::endl;
    std::cout << "Type 'ctrl+c' or enter invalid inputs to exit." << std::endl;
    std::cout << "==================================================" << std::endl;

    double inputX, inputY;
    std::vector<std::shared_ptr<Value>> userInputs(2);

    // Keep prompting the user for coordinate pairs
    while (true)
    {
        std::cout << "\nEnter coordinates (X Y): ";
        if (!(std::cin >> inputX >> inputY))
        {
            std::cout << "Exiting interactive mode. Goodbye!" << std::endl;
            break;
        }

        // 1. Pack the raw coordinates into your custom Value nodes
        userInputs[0] = std::make_shared<Value>(inputX);
        userInputs[1] = std::make_shared<Value>(inputY);

        // 2. Pass them through your trained MLP forward graph
        auto outputVec = mlp.predict(userInputs);
        double prediction = outputVec[0]->GetValue();

        // 3. Print the results and translate what they mean
        std::cout << "--> Network Prediction raw output: " << prediction << std::endl;
        
        if (prediction >= 0.5)
        {
            std::cout << "--> Classification: POSITIVE (Targets 1.0 - Q1/Q3)" << std::endl;
        }
        else
        {
            std::cout << "--> Classification: NEGATIVE (Targets 0.0 - Q2/Q4)" << std::endl;
        }
    }

}