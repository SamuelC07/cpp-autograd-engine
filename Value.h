#pragma once

#include <iostream>
#include <memory>
#include <vector>
#include <unordered_set>
#include <functional>
#include <algorithm>
#include <cmath>

class Value : public std::enable_shared_from_this<Value>
{
private:
    double m_num;
    double m_gradient = 0.0;
    std::vector<std::shared_ptr<Value>> m_children;
    std::function<void()> _backwards = [](){};

public:
    Value(double num) : m_num(num) {}

    double GetValue() const
    {
        return m_num;
    }

    double GetGradient() const
    {
        return m_gradient;
    }

    // print children
    void PrintChildren() const
    {
        for (int i = 0; i < m_children.size(); i++)
        {
            std::cout << m_children.at(i) << ", ";
        }
        std:: cout << std::endl;
        
    }


    // return children
    std::vector<std::shared_ptr<Value>> GetChildren() const
    {
        return m_children;
    }

    // overload add
    friend std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs)
    {
        std::shared_ptr<Value> out = std::make_shared<Value>(lhs->m_num + rhs->m_num);
        out->m_children = {lhs, rhs};
        std::weak_ptr<Value> weak_out = out;

        out->_backwards = [lhs, rhs, weak_out]()
        {
            auto out = weak_out.lock();
            lhs->m_gradient += out->m_gradient;
            rhs->m_gradient += out->m_gradient;
        };

        return out;
    }

    // overload multiply
    friend std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs)
    {
        std::shared_ptr<Value> out = std::make_shared<Value>(lhs->m_num * rhs->m_num);
        out->m_children = {lhs, rhs};
        std::weak_ptr<Value> weak_out = out;

        out->_backwards = [lhs, rhs, weak_out]()
        {
            auto out = weak_out.lock();
            lhs->m_gradient += rhs->m_num * out->m_gradient;
            rhs->m_gradient += lhs->m_num * out->m_gradient;
        };

        return out;
    }
    
    // add pow operator
    friend std::shared_ptr<Value> pow(const std::shared_ptr<Value>& base, int exp)
    {
        std::shared_ptr<Value> out = std::make_shared<Value>(std::pow(base->m_num, exp));
        out->m_children = { base };
        std::weak_ptr<Value> weak_out = out;

        out->_backwards = [base, weak_out, exp ]()
        {
            auto out = weak_out.lock();
            base->m_gradient += exp * std::pow(base->m_num, exp - 1) * out->m_gradient;
        };
        return out;
    }

    // add subtract operator
    friend std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs)
    {
        return lhs + (std::make_shared<Value>(-1.0) * rhs);
    }

    // overload the << operator
    friend std::ostream& operator<<(std::ostream& os, const Value& val)
    {
        return os << val.m_num;
    }

    // overload for pointers
    friend std::ostream& operator<<(std::ostream& os, const std::shared_ptr<Value>& val)
    {
        return os << val->m_num;
    }

    // ReLU squeeze function
    std::shared_ptr<Value> ReLU()
    {
        auto out = std::make_shared<Value>(std::max(0.0, m_num));
        out->m_children = { shared_from_this() };
        std::weak_ptr<Value> weak_out = out;
        auto self = shared_from_this();

        out->_backwards = [self, weak_out]()
        {
            auto out = weak_out.lock();
            if (self->m_num > 0)
            {
                self->m_gradient += out->m_gradient;
            }
        };
        return out;
    }

    // leaky ReLU function for small inputs
    std::shared_ptr<Value> LeakyReLU()
    {
        double val = m_num > 0.0 ? m_num : 0.01 * m_num;
        auto out = std::make_shared<Value>(val);
        
        out->m_children = { shared_from_this() };
        std::weak_ptr<Value> weak_out = out;
        auto self = shared_from_this();

        out->_backwards = [self, weak_out]()
        {
            if (auto out = weak_out.lock())
            {
                if (self->m_num > 0.0)
                {
                    self->m_gradient += out->m_gradient;
                }
                else
                {
                    self->m_gradient += 0.01 * out->m_gradient;
                }
            }
        };
        return out;
    }

    // backwards propogation
    void backwards()
    {
        std::vector<std::shared_ptr<Value>> sorted;
        std::unordered_set<std::shared_ptr<Value>> visited;
        // build a topological sorted list of all values
        std::function<void(std::shared_ptr<Value>)> topoSort = [&](const std::shared_ptr<Value>& v)
        {
            if (!visited.contains(v))
            {
                visited.insert(v);
                for (std::shared_ptr<Value> child : v->m_children)
                {
                    topoSort(child);
                }
                sorted.push_back(v);
            }
        };
        topoSort(shared_from_this());

        this->m_gradient = 1.0;

        // go backwards through the list and do their backwards functions.
        for (auto i = sorted.rbegin(); i != sorted.rend(); ++i)
        {
            (*i)->_backwards();
        }
    }

    void printTopo()
    {
        std::vector<std::shared_ptr<Value>> sorted;
        std::unordered_set<std::shared_ptr<Value>> visited;
        // build a topological sorted list of all values
        std::function<void(std::shared_ptr<Value>)> topoSort = [&](std::shared_ptr<Value> v)
        {
            if (!visited.contains(v))
            {
                visited.insert(v);
                for (std::shared_ptr<Value> child : v->m_children)
                {
                    topoSort(child);
                }
                sorted.push_back(v);
            }
        };
        topoSort(shared_from_this());

        for (auto i = sorted.begin(); i != sorted.end(); ++i)
        {
            std::cout << (*i) << " ";
        }
        std::cout << std::endl;
    }

    void reset()
    {
        m_gradient = 0;
    }

    void addNum(double num)
    {
        m_num += num;
    }

    double getGradient() const
    {
        return m_gradient;
    }

};