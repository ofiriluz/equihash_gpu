/**
 * @file proof.cpp
 * @author ofir iluz (iluzofir@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2018-11-26
 * 
 * @copyright Copyright (c) 2018
 * 
 */

#include "equihash_gpu/equihash/proof.h"

namespace Equihash
{
    Proof::Proof():solution_nonce_(0)
    {

    }

    Proof::Proof(const std::vector<uint32_t> & solution_inputs, uint32_t solution_nonce)
    {

    }

    Proof::~Proof()
    {

    }

    std::vector<uint32_t> & Proof::get_solution_inputs()
    {
        return solution_inputs_;
    }

    uint32_t Proof::get_solution_nonce()const
    {
        return solution_nonce_;
    }
}